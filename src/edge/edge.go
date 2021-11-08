package edge

import (
	"context"
	"errors"
	"sync"
	pe "vakd/proto/edge"
	ps "vakd/proto/source"
	pw "vakd/proto/worker"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

var (
	ErrSourceNotFound = errors.New("source ID not found")
)

type Frame struct {
	id      int
	content []byte
}

type Stream struct {
	id        int
	received  int
	processed int
	buffer    []*Frame
	client    ps.SourceForEdgeClient
}

type Edge struct {
	pe.EdgeForSourceServer
	index   int
	streams []*Stream
	client  pw.WorkerForEdgeClient
	m       sync.RWMutex
}

func NewEdge(client pw.WorkerForEdgeClient) (*Edge, error) {
	edge := &Edge{
		index:   0,
		streams: make([]*Stream, 0),
		client:  client,
		m:       sync.RWMutex{},
	}
	go edge.inferenceLoop()
	return edge, nil
}

func (e *Edge) inferenceLoop() {

}

func (e *Edge) Register(ctx context.Context, request *pe.RegisterRequest) (*pe.RegisterResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	client := ps.NewSourceForEdgeClient(connection)
	e.m.Lock()
	stream := &Stream{
		id:        e.index,
		received:  -1,
		processed: -1,
		buffer:    make([]*Frame, 0),
		client:    client,
	}
	e.index++
	e.streams = append(e.streams, stream)
	e.m.Unlock()
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
	e.m.RLock()
	_, i := e.findStream(id)
	e.m.RUnlock()
	if i < 0 {
		return nil, ErrSourceNotFound
	}
	e.m.Lock()
	e.streams = append(e.streams[:i], e.streams[i+1:]...)
	e.m.Unlock()
	logrus.Infof("Disconnected with source %d", id)
	return &pe.TerminateResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SendFrameRequest) (*pe.SendFrameResponse, error) {
	id := int(request.Id)
	e.m.RLock()
	stream, i := e.findStream(id)
	e.m.RUnlock()
	if i < 0 {
		return nil, ErrSourceNotFound
	}
	e.m.Lock()
	stream.buffer = append(stream.buffer, &Frame{
		id:      int(request.Current),
		content: request.Content,
	})
	logrus.Infof("Received frame %d from stream %d", request.Current, request.Id)
	stream.received++
	e.m.Unlock()
	return &pe.SendFrameResponse{}, nil
}
