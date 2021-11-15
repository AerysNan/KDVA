package edge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"
	pe "vakd/proto/edge"
	ps "vakd/proto/source"
	pw "vakd/proto/worker"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

var (
	ErrSourceNotFound = errors.New("source ID not found")
)

const MONITOR_INTERVAL = time.Millisecond * 1000

type Frame struct {
	id      int
	source  int
	content []byte
}

type Stream struct {
	m         sync.RWMutex
	id        int
	received  int
	processed int
	start     time.Time
	client    ps.SourceForEdgeClient
}

type Edge struct {
	pe.EdgeForSourceServer
	m       sync.RWMutex
	index   int
	streams []*Stream
	client  pw.WorkerForEdgeClient
	buffer  chan *Frame
}

func NewEdge(client pw.WorkerForEdgeClient) (*Edge, error) {
	edge := &Edge{
		index:   0,
		streams: make([]*Stream, 0),
		client:  client,
		m:       sync.RWMutex{},
		buffer:  make(chan *Frame, 100),
	}
	go edge.inferenceLoop()
	go edge.monitorLoop()
	return edge, nil
}

func (e *Edge) monitorLoop() {
	timer := time.NewTicker(MONITOR_INTERVAL)
	for {
		func() {
			e.m.RLock()
			defer e.m.RUnlock()
			if len(e.streams) == 0 {
				return
			}
			throughput := 0.0
			maxDelay := -1
			for _, stream := range e.streams {
				stream.m.RLock()
				fps := float64(stream.processed) / time.Since(stream.start).Seconds()
				if stream.received-stream.processed > maxDelay {
					maxDelay = stream.received - stream.processed
				}
				throughput += fps
				stream.m.RUnlock()
			}
			logrus.Infof("Current throughput: %.2f fps; Max delay: %d frames", throughput, maxDelay)
		}()
		<-timer.C
	}
}
func (e *Edge) inferenceLoop() {
	for {
		func(frame *Frame) {
			e.m.RLock()
			defer e.m.RUnlock()
			stream, i := e.findStream(frame.source)
			if i < 0 {
				logrus.WithError(ErrSourceNotFound).Errorf("Inference source %d not found", frame.source)
				return
			}
			response, err := e.client.Infer(context.Background(), &pw.InferRequest{
				Content: frame.content,
			})
			if err != nil {
				logrus.WithError(err).Errorf("Infer frame %d on source %d failed", frame.id, frame.source)
				return
			}
			bytes := e.responseToJSON(response)
			file, err := os.OpenFile(fmt.Sprintf("result/%d_%d.json", frame.source, frame.id), os.O_CREATE|os.O_WRONLY, 0777)
			if err != nil {
				logrus.WithError(err).Errorf("Open dump file for frame %d on source %d failed", frame.id, frame.source)
				return
			}
			defer file.Close()
			if _, err = file.Write(bytes); err != nil {
				logrus.WithError(err).Errorf("Dump inference result for frame %d on source %d failed", frame.id, frame.source)
				return
			}
			stream.m.Lock()
			defer stream.m.Unlock()
			stream.processed++
		}(<-e.buffer)
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

func (e *Edge) Register(ctx context.Context, request *pe.RegisterRequest) (*pe.RegisterResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	client := ps.NewSourceForEdgeClient(connection)
	e.m.Lock()
	defer e.m.Unlock()
	stream := &Stream{
		m:         sync.RWMutex{},
		id:        e.index,
		start:     time.Now(),
		received:  0,
		processed: 0,
		client:    client,
	}
	e.index++
	e.streams = append(e.streams, stream)
	logrus.Infof("Connected with source %d", stream.id)
	return &pe.RegisterResponse{
		Id: int64(stream.id),
	}, nil
}

func (e *Edge) findStream(id int) (*Stream, int) {
	for i, stream := range e.streams {
		if stream.id == id {
			return stream, i
		}
	}
	return nil, -1
}

func (e *Edge) Terminate(ctx context.Context, request *pe.TerminateRequest) (*pe.TerminateResponse, error) {
	id := int(request.Id)
	e.m.Lock()
	defer e.m.Unlock()
	_, i := e.findStream(id)
	if i < 0 {
		return nil, ErrSourceNotFound
	}
	e.streams = append(e.streams[:i], e.streams[i+1:]...)
	logrus.Infof("Disconnected with source %d", id)
	return &pe.TerminateResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SendFrameRequest) (*pe.SendFrameResponse, error) {
	id := int(request.Id)
	e.m.RLock()
	defer e.m.RUnlock()
	stream, i := e.findStream(id)
	if i < 0 {
		return nil, ErrSourceNotFound
	}
	e.buffer <- &Frame{
		id:      int(request.Current),
		source:  id,
		content: request.Content,
	}
	logrus.Debugf("Received frame %d from stream %d", request.Current, id)
	stream.m.Lock()
	defer stream.m.Unlock()
	stream.received++
	return &pe.SendFrameResponse{}, nil
}
