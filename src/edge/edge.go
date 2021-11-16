package edge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
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
	ORINGAL_FRAMERATE = 30
	MONITOR_INTERVAL  = time.Millisecond * 1000
)

type Frame struct {
	index   int
	source  int
	content []byte
}

type Source struct {
	m         sync.RWMutex
	id        int
	uploadFPS int
	received  int
	processed int
	disabled  bool
	start     time.Time
	client    ps.SourceForEdgeClient
}

type Edge struct {
	pe.EdgeForSourceServer
	pe.EdgeForCloudServer
	m            sync.RWMutex
	id           int
	index        int
	address      string
	sources      map[int]*Source
	inferBuffer  chan *Frame
	uploadBuffer chan *Frame
	workerClient pw.WorkerForEdgeClient
	cloudClient  pc.CloudForEdgeClient
}

func NewEdge(address string, workerClient pw.WorkerForEdgeClient, cloudClient pc.CloudForEdgeClient) (*Edge, error) {
	edge := &Edge{
		m:            sync.RWMutex{},
		index:        0,
		address:      address,
		sources:      make(map[int]*Source),
		inferBuffer:  make(chan *Frame),
		uploadBuffer: make(chan *Frame),
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
			e.m.RLock()
			defer e.m.RUnlock()
			throughput := 0.0
			maxDelay := -1
			count := 0
			for _, source := range e.sources {
				if source.disabled {
					continue
				}
				count++
				source.m.RLock()
				fps := float64(source.processed) / time.Since(source.start).Seconds()
				if source.received-source.processed > maxDelay {
					maxDelay = source.received - source.processed
				}
				throughput += fps
				source.m.RUnlock()
			}
			if count != 0 {
				logrus.Infof("Source: %d; Throughput: %.2f fps; Delay: %d frames", count, throughput, maxDelay)
			}
		}()
		<-timer.C
	}
}

func (e *Edge) uploadLoop() {
	for {
		frame := <-e.uploadBuffer
		_, err := e.cloudClient.UploadFrame(context.Background(), &pc.UploadFrameRequest{
			Edge:    int64(e.id),
			Source:  int64(frame.source),
			Index:   int64(frame.index),
			Content: frame.content,
		})
		if err != nil {
			logrus.WithError(err).Errorf("Upload frame %d of source %d failed", frame.index, frame.source)
		}
	}
}

func (e *Edge) inferenceLoop() {
	for {
		func(frame *Frame) {
			e.m.RLock()
			defer e.m.RUnlock()
			source, ok := e.sources[frame.source]
			if !ok {
				logrus.WithError(ErrSourceNotFound).Errorf("Inference source %d not found", frame.source)
				return
			}
			response, err := e.workerClient.Infer(context.Background(), &pw.InferRequest{
				Source:  int64(source.id),
				Content: frame.content,
			})
			if err != nil {
				logrus.WithError(err).Errorf("Infer frame %d on source %d failed", frame.index, frame.source)
				return
			}
			bytes := e.responseToJSON(response)
			file, err := os.OpenFile(fmt.Sprintf("dump/result/%d_%d/%d.json", e.id, frame.source, frame.index), os.O_CREATE|os.O_WRONLY, 0777)
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
			defer source.m.Unlock()
			source.processed++
			if frame.index%(ORINGAL_FRAMERATE/source.uploadFPS) == 0 {
				go func() {
					e.uploadBuffer <- frame
				}()
			}
		}(<-e.inferBuffer)
	}
}

func (e *Edge) responseToJSON(response *pw.InferResponse) []byte {
	results := make([][][]float64, 0)
	for _, classResult := range response.Result {
		boxes := make([][]float64, 0)
		for _, box := range classResult.Boxes {
			boxes = append(boxes, box.Params)
		}
		results = append(results, boxes)
	}
	bytes, err := json.Marshal(results)
	if err != nil {
		logrus.WithError(err).Error("Marshal inference result to JSON failed")
		return nil
	}
	return bytes
}

func (e *Edge) AddSource(ctx context.Context, request *pe.AddSourceRequest) (*pe.AddSourceResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	client := ps.NewSourceForEdgeClient(connection)
	e.m.Lock()
	defer e.m.Unlock()
	source := &Source{
		m:         sync.RWMutex{},
		id:        e.index,
		disabled:  false,
		uploadFPS: 1,
		received:  0,
		processed: 0,
		client:    client,
	}
	if _, err = e.workerClient.AddModel(context.Background(), &pw.AddModelRequest{
		Source: int64(source.id),
	}); err != nil {
		return nil, err
	}
	e.index++
	source.start = time.Now()
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
	// if _, err := e.workerClient.RemoveModel(context.Background(), &pw.RemoveModelRequest{
	// 	Source: int64(id),
	// }); err != nil {
	// 	return nil, err
	// }
	source.disabled = true
	logrus.Infof("Disconnected with source %d", id)
	return &pe.RemoveSourceResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SendFrameRequest) (*pe.SendFrameResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	defer e.m.RUnlock()
	source, ok := e.sources[id]
	if !ok {
		return nil, ErrSourceNotFound
	}
	go func() {
		e.inferBuffer <- &Frame{
			index:   int(request.Index),
			source:  id,
			content: request.Content,
		}
	}()
	logrus.Debugf("Received frame %d from stream %d", request.Index, id)
	source.m.Lock()
	defer source.m.Unlock()
	source.received++
	return &pe.SendFrameResponse{}, nil
}

func (e *Edge) LoadModel(ctx context.Context, request *pe.LoadModelRequest) (*pe.LoadModelResponse, error) {
	id := int(request.Source)
	if _, ok := e.sources[id]; !ok {
		return nil, ErrSourceNotFound
	}
	path := fmt.Sprintf("dump/model/%d_%d/%d.pth", e.id, request.Source, request.Epoch)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0777)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	_, err = file.Write(request.Model)
	if err != nil {
		return nil, err
	}
	_, err = e.workerClient.UpdateModel(context.Background(), &pw.UpdateModelRequest{
		Source: request.Source,
		Path:   path,
	})
	if err != nil {
		return nil, err
	}
	return &pe.LoadModelResponse{}, nil
}
