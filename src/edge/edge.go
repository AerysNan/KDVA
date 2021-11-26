package edge

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"
	pc "vakd/proto/cloud"

	pe "vakd/proto/edge"
	ps "vakd/proto/source"
	pw "vakd/proto/worker"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

var (
	ErrSourceNotFound = errors.New("source ID not found")
)

const (
	MONITOR_INTERVAL = time.Millisecond * 1000
)

type Profile struct {
	accuracy float64
}

type Frame struct {
	index   int
	source  int
	content []byte
}

type Pair struct {
	frame      *Frame
	annotation []byte
}

type Source struct {
	id             int
	weight         int
	received       int
	processed      int
	currentFPS     int
	originalFPS    int
	uploadInterval int
	disabled       bool
	dataset        string
	profiles       []*Profile
	start          time.Time
	buffer         chan *Frame
	client         ps.SourceForEdgeClient
}

type Edge struct {
	pe.EdgeForSourceServer
	pe.EdgeForCloudServer
	m            sync.RWMutex
	id           int
	index        int
	address      string
	throughput   float64
	uploadDict   map[int]int
	sources      map[int]*Source
	uploadBuffer chan *Pair
	workerClient pw.WorkerForEdgeClient
	cloudClient  pc.CloudForEdgeClient
}

func NewEdge(address string, workerClient pw.WorkerForEdgeClient, cloudClient pc.CloudForEdgeClient) (*Edge, error) {
	edge := &Edge{
		m:            sync.RWMutex{},
		index:        0,
		address:      address,
		sources:      make(map[int]*Source),
		throughput:   100,
		uploadBuffer: make(chan *Pair),
		uploadDict:   make(map[int]int),
		workerClient: workerClient,
		cloudClient:  cloudClient,
	}
	response, err := cloudClient.AddEdge(context.Background(), &pc.AddEdgeRequest{
		Address: address,
	})
	if err != nil {
		return nil, err
	}
	edge.id = int(response.Id)
	go edge.inferenceLoop()
	go edge.monitorLoop()
	go edge.uploadLoop()
	return edge, nil
}

func (e *Edge) monitorLoop() {
	timer := time.NewTicker(MONITOR_INTERVAL)
	delay := -1
	for {
		func() {
			throughput := 0.0
			maxDelay := -1
			count := 0
			for _, source := range e.sources {
				if source.disabled {
					continue
				}
				count++
				fps := float64(source.processed) / time.Since(source.start).Seconds()
				if source.received-source.processed > maxDelay {
					maxDelay = source.received - source.processed
				}
				throughput += fps
			}
			if count != 0 {
				logrus.Infof("Source: %d; Throughput: %.2f fps; Delay: %d frames", count, throughput, maxDelay)
				totalFPS := 0
				e.m.Lock()
				for _, source := range e.sources {
					totalFPS += source.currentFPS
				}
				e.m.Unlock()
				if throughput > 0 {
					e.throughput = throughput
				}
				if math.Abs(float64(totalFPS)-e.throughput) > 1 || maxDelay > delay {
					delay = maxDelay
					e.adjustFramerate()
				}
			}
		}()
		<-timer.C
	}
}

func (e *Edge) adjustFramerate() {
	e.m.RLock()
	defer e.m.RUnlock()
	totalWeight, throughput := 0, e.throughput
	for _, source := range e.sources {
		totalWeight += source.weight
	}
	for id, source := range e.sources {
		framerate := math.Round(throughput * float64(source.weight) / float64(totalWeight))
		if _, err := source.client.SetFramerate(context.Background(), &ps.SetFramerateRequest{
			FrameRate: int64(framerate),
		}); err != nil {
			logrus.WithError(err).Errorf("Change framerate for source %d failed", id)
		} else {
			logrus.Infof("Change framerate for source %d to %.2f fps", id, framerate)
			source.currentFPS = int(framerate)
		}
	}
}

func (e *Edge) uploadLoop() {
	for {
		pair := <-e.uploadBuffer
		source, ok := e.sources[pair.frame.source]
		if !ok {
			logrus.WithError(ErrSourceNotFound).Errorf("Upload frame %d of source %d failed", pair.frame.index, pair.frame.source)
			continue
		}
		if _, err := e.cloudClient.SendFrame(context.Background(), &pc.EdgeSendFrameRequest{
			Edge:       int64(e.id),
			Source:     int64(pair.frame.source),
			Index:      int64(pair.frame.index),
			Dataset:    source.dataset,
			Content:    pair.frame.content,
			Annotation: pair.annotation,
		}); err != nil {
			logrus.WithError(err).Errorf("Upload frame %d of source %d failed", pair.frame.index, pair.frame.source)
		} else {
			logrus.Debugf("Upload frame %d of source %d", pair.frame.index, pair.frame.source)
		}
	}
}

func (e *Edge) inferenceLoop() {
	for {
		keys := make([]int, 0)
		maxWeight := 0
		e.m.RLock()
		for key, source := range e.sources {
			keys = append(keys, key)
			if source.weight > maxWeight {
				maxWeight = source.weight
			}
		}
		e.m.RUnlock()
		sort.Ints(keys)
		for w := 1; w <= maxWeight; w++ {
			for _, key := range keys {
				if e.sources[key].weight < w {
					continue
				}
				func(frame *Frame) {
					source, ok := e.sources[frame.source]
					if !ok {
						logrus.WithError(ErrSourceNotFound).Errorf("Inference source %d not found", frame.source)
						return
					}
					response, err := e.workerClient.InferFrame(context.Background(), &pw.InferFrameRequest{
						Source:  int64(source.id),
						Content: frame.content,
					})
					logrus.Debugf("Infer frame %d on source %d", frame.index, frame.source)
					if err != nil {
						logrus.WithError(err).Errorf("Infer frame %d on source %d failed", frame.index, frame.source)
						return
					}
					bytes := response.Result
					file, err := os.OpenFile(fmt.Sprintf("dump/result/%d_%d/%d.pkl", e.id, frame.source, frame.index), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
					if err != nil {
						logrus.WithError(err).Errorf("Open dump file for frame %d on source %d failed", frame.index, frame.source)
						return
					}
					defer file.Close()
					if _, err = file.Write(bytes); err != nil {
						logrus.WithError(err).Errorf("Dump inference result for frame %d on source %d failed", frame.index, frame.source)
						return
					}
					source.processed++
					if frame.index/source.uploadInterval > e.uploadDict[frame.source] {
						e.uploadDict[frame.source]++
						go func() {
							logrus.Debugf("Prepare upload frame %d from stream %d", frame.index, frame.source)
							e.uploadBuffer <- &Pair{
								frame:      frame,
								annotation: bytes,
							}
						}()
					}
				}(<-e.sources[key].buffer)
			}
		}
	}
}

func (e *Edge) AddSource(ctx context.Context, request *pe.AddSourceRequest) (*pe.AddSourceResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	client := ps.NewSourceForEdgeClient(connection)
	e.m.Lock()
	id := e.index
	e.index++
	e.m.Unlock()
	source := &Source{
		id:             id,
		weight:         1,
		received:       0,
		processed:      0,
		uploadInterval: 100,
		disabled:       false,
		dataset:        request.Dataset,
		profiles:       make([]*Profile, 0),
		currentFPS:     int(request.Fps),
		originalFPS:    int(request.Fps),
		client:         client,
		buffer:         make(chan *Frame),
	}
	if _, err = e.workerClient.AddModel(context.Background(), &pw.AddModelRequest{
		Source: int64(source.id),
	}); err != nil {
		return nil, err
	}
	logrus.Debugf("Model added for source %d", source.id)
	e.m.Lock()
	defer e.m.Unlock()
	source.start = time.Now()
	e.uploadDict[source.id] = -1
	e.sources[source.id] = source
	logrus.Infof("Connected with source %d", source.id)
	os.MkdirAll(fmt.Sprintf("dump/result/%d_%d", e.id, source.id), 0777)
	os.MkdirAll(fmt.Sprintf("dump/model/%d_%d", e.id, source.id), 0777)
	return &pe.AddSourceResponse{
		Id: int64(source.id),
	}, nil
}

func (e *Edge) RemoveSource(ctx context.Context, request *pe.RemoveSourceRequest) (*pe.RemoveSourceResponse, error) {
	id := int(request.Id)
	e.m.Lock()
	defer e.m.Unlock()
	source, ok := e.sources[id]
	if !ok {
		return nil, ErrSourceNotFound
	}
	source.disabled = true
	logrus.Infof("Disconnected with source %d", id)
	return &pe.RemoveSourceResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SourceSendFrameRequest) (*pe.SourceSendFrameResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	defer e.m.RUnlock()
	source, ok := e.sources[id]
	if !ok {
		return nil, ErrSourceNotFound
	}
	go func() {
		source.buffer <- &Frame{
			index:   int(request.Index),
			source:  id,
			content: request.Content,
		}
	}()
	logrus.Debugf("Received frame %d from stream %d", request.Index, id)
	source.received++
	return &pe.SourceSendFrameResponse{}, nil
}

func (e *Edge) UpdateModel(ctx context.Context, request *pe.CloudUpdateModelRequest) (*pe.CloudUpdateModelResponse, error) {
	id := int(request.Source)
	if _, ok := e.sources[id]; !ok {
		return nil, ErrSourceNotFound
	}
	path := fmt.Sprintf("dump/model/%d_%d/%d.pth", e.id, request.Source, request.Epoch)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	_, err = file.Write(request.Model)
	if err != nil {
		return nil, err
	}
	_, err = e.workerClient.UpdateModel(context.Background(), &pw.EdgeUpdateModelRequest{
		Source: request.Source,
		Path:   path,
	})
	if err != nil {
		return nil, err
	}
	return &pe.CloudUpdateModelResponse{}, nil
}

func (e *Edge) ReportProfile(ctx context.Context, request *pe.CloudReportProfileRequest) (*pe.CloudReportProfileResponse, error) {
	source, ok := e.sources[int(request.Source)]
	if !ok {
		logrus.WithError(ErrSourceNotFound).Errorf("Report profile for source %d failed", request.Source)
		return &pe.CloudReportProfileResponse{}, nil
	}
	logrus.Infof("Receive profile for source %d accuracy %.2f", request.Source, request.Accuracy)
	accuracy := request.Accuracy
	if accuracy < 0 {
		if len(source.profiles) == 0 {
			accuracy = 0.3
		} else {
			sum := 0.0
			for _, profile := range source.profiles {
				sum += profile.accuracy
			}
			accuracy = sum / float64(len(source.profiles))
		}
	}
	source.profiles = append(source.profiles, &Profile{
		accuracy: accuracy,
	})
	return &pe.CloudReportProfileResponse{}, nil
}
