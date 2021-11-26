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
	"vakd/util"

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
	TIMEOUT_INTERVAL = time.Millisecond * 1000
	FIXED_THROUGHPUT = 20
)

const (
	SOURCE_STATUS_CONNECTED = iota
	SOURCE_STATUS_DISCONNECTED
	SOURCE_STATUS_CLOSED
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
	m              sync.RWMutex
	id             int
	weight         float64
	received       int
	processed      int
	receivedAcc    int
	processedAcc   int
	currentFPS     int
	uploadInterval int
	status         int
	updated        bool
	dataset        string
	lastMonitor    time.Time
	profiles       []*Profile
	buffer         chan *Frame
	closeCh        chan struct{}
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
	allocator    Allocator
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
		allocator:    &EvenAllocator{},
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
	for {
		func() {
			throughput, maxDelay, count, totalFPS := 0.0, 0, 0, 0
			ids := make([]int, 0)
			e.m.RLock()
			for id, source := range e.sources {
				if source.status != SOURCE_STATUS_CLOSED {
					ids = append(ids, id)
				}
			}
			e.m.RUnlock()
			for _, id := range ids {
				source := e.sources[id]
				count++
				fps := float64(source.processed) / time.Since(source.lastMonitor).Seconds()
				if source.receivedAcc-source.processedAcc > maxDelay {
					maxDelay = source.receivedAcc - source.processedAcc
				}
				source.m.Lock()
				source.received = 0
				source.processed = 0
				source.m.Unlock()
				throughput += fps
				source.lastMonitor = time.Now()
				totalFPS += source.currentFPS
			}
			logrus.Infof("Source: %d; Throughput: %.2f fps; FPS: %d; Delay: %d frames", count, throughput, totalFPS, maxDelay)
			e.throughput = throughput
			e.updateWeightAndFramerate()
		}()
		<-timer.C
	}
}

func (e *Edge) updateWeightAndFramerate() {
	totalWeight, throughput := 0.0, FIXED_THROUGHPUT
	ids := make([]int, 0)
	e.m.RLock()
	e.allocator.Allocate()
	for id, source := range e.sources {
		if source.status != SOURCE_STATUS_CONNECTED {
			continue
		}
		totalWeight += source.weight
		ids = append(ids, id)
	}
	e.m.RUnlock()
	sort.Ints(ids)
	framerateMap, remainder := make(map[int]int), int(throughput)
	for _, id := range ids {
		framerateMap[id] = int(float64(throughput) * e.sources[id].weight / totalWeight)
		remainder -= framerateMap[id]
	}
	for _, id := range ids {
		if remainder == 0 {
			break
		}
		framerateMap[id]++
		remainder--
	}
	for _, id := range ids {
		if framerateMap[id] == e.sources[id].currentFPS {
			continue
		}
		if _, err := e.sources[id].client.SetFramerate(context.Background(), &ps.SetFramerateRequest{
			FrameRate: int64(framerateMap[id]),
		}); err != nil {
			logrus.WithError(err).Errorf("Change framerate for source %d failed", id)
		} else {
			e.sources[id].currentFPS = framerateMap[id]
			logrus.Infof("Change framerate for source %d to %d fps", id, e.sources[id].currentFPS)
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
		ids, weightMap := make([]int, 0), make(map[int]int)
		maxFPS := 0
		e.m.RLock()
		for id, source := range e.sources {
			if len(source.buffer) == 0 && source.status == SOURCE_STATUS_DISCONNECTED {
				source.status = SOURCE_STATUS_CLOSED
				source.closeCh <- struct{}{}
			}
			if source.status == SOURCE_STATUS_CLOSED {
				continue
			}
			ids = append(ids, id)
			source.m.RLock()
			if source.currentFPS > maxFPS {
				maxFPS = source.currentFPS
			}
			weightMap[id] = source.currentFPS
			source.m.RUnlock()
		}
		e.m.RUnlock()
		if len(ids) == 0 {
			continue
		}
		sort.Ints(ids)
		gcd := util.Reduce(weightMap)
		for w := 1; w <= maxFPS/gcd; w++ {
			for _, id := range ids {
				if weightMap[id] < w {
					continue
				}
				timeoutTimer := time.NewTimer(TIMEOUT_INTERVAL)
				select {
				case frame := <-e.sources[id].buffer:
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
						source.m.Lock()
						source.processed++
						source.processedAcc++
						source.m.Unlock()
						if frame.index/source.uploadInterval > e.uploadDict[frame.source] {
							e.uploadDict[frame.source] = frame.index / source.uploadInterval
							go func(p *Pair) {
								e.uploadBuffer <- p
							}(&Pair{
								frame:      frame,
								annotation: bytes,
							})
						}
					}(frame)
				case <-timeoutTimer.C:
					logrus.Warnf("Source %d slow, skip round robin", id)
				}
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
		m:              sync.RWMutex{},
		id:             id,
		weight:         1,
		received:       0,
		processed:      0,
		receivedAcc:    0,
		processedAcc:   0,
		uploadInterval: 100,
		status:         SOURCE_STATUS_CONNECTED,
		updated:        false,
		dataset:        request.Dataset,
		profiles:       make([]*Profile, 0),
		currentFPS:     int(request.Fps),
		client:         client,
		closeCh:        make(chan struct{}),
		buffer:         make(chan *Frame),
	}
	if _, err = e.workerClient.AddModel(context.Background(), &pw.AddModelRequest{
		Source: int64(source.id),
	}); err != nil {
		return nil, err
	}
	logrus.Debugf("Model added for source %d", source.id)
	e.m.Lock()
	e.uploadDict[source.id] = -1
	e.sources[source.id] = source
	e.allocator.Reset(e.sources)
	defer e.m.Unlock()
	logrus.Infof("Connected with source %d", source.id)
	os.MkdirAll(fmt.Sprintf("dump/result/%d_%d", e.id, source.id), 0777)
	os.MkdirAll(fmt.Sprintf("dump/model/%d_%d", e.id, source.id), 0777)
	source.lastMonitor = time.Now()
	return &pe.AddSourceResponse{
		Id:   int64(source.id),
		Edge: int64(e.id),
	}, nil
}

func (e *Edge) RemoveSource(ctx context.Context, request *pe.RemoveSourceRequest) (*pe.RemoveSourceResponse, error) {
	id := int(request.Id)
	source, ok := e.sources[id]
	if !ok {
		return nil, ErrSourceNotFound
	}
	source.status = SOURCE_STATUS_DISCONNECTED
	logrus.Infof("Disconnected with source %d", id)
	<-source.closeCh
	logrus.Infof("Source %d closed", id)
	return &pe.RemoveSourceResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SourceSendFrameRequest) (*pe.SourceSendFrameResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	source, ok := e.sources[id]
	e.m.RUnlock()
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
	logrus.Debugf("Receive frame %d from stream %d", request.Index, id)
	source.m.Lock()
	source.received++
	source.receivedAcc++
	source.m.Unlock()
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
	e.m.RLock()
	source, ok := e.sources[int(request.Source)]
	e.m.RUnlock()
	if !ok {
		logrus.WithError(ErrSourceNotFound).Errorf("Report profile for source %d failed", request.Source)
		return &pe.CloudReportProfileResponse{}, nil
	}
	logrus.Infof("Receive profile for source %d in range [%d, %d] with accuracy %.2f", request.Source, request.Begin, request.End, request.Accuracy)
	source.m.Lock()
	source.profiles = append(source.profiles, &Profile{
		accuracy: math.Max(request.Accuracy, 0),
	})
	source.updated = true
	source.m.Unlock()
	return &pe.CloudReportProfileResponse{}, nil
}
