package cloud

import (
	"context"
	"errors"
	"sync"
	"time"

	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pt "vakd/proto/trainer"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

var (
	ErrEdgeNotFound = errors.New("edge ID not found")
)

const (
	MONITOR_INTERVAL = time.Millisecond * 1000
	EPOCH_LENGTH     = 500
)

type Edge struct {
	m        sync.RWMutex
	id       int
	disabled bool
	client   pe.EdgeForCloudClient
}

type Cloud struct {
	pc.CloudForEdgeServer
	pc.CloudForTrainerServer
	m       sync.RWMutex
	index   int
	address string
	edges   map[int]*Edge
	client  pt.TrainerForCloudClient
}

func NewCloud(address string, client pt.TrainerForCloudClient) (*Cloud, error) {
	cloud := &Cloud{
		m:       sync.RWMutex{},
		index:   0,
		address: address,
		edges:   make(map[int]*Edge),
		client:  client,
	}
	return cloud, nil
}

func (c *Cloud) AddEdge(ctx context.Context, request *pc.AddEdgeRequest) (*pc.AddEdgeResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure(), grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(1<<30), grpc.MaxCallRecvMsgSize(1<<30)))
	if err != nil {
		return nil, err
	}
	client := pe.NewEdgeForCloudClient(connection)
	c.m.Lock()
	defer c.m.Unlock()
	edge := &Edge{
		m:        sync.RWMutex{},
		id:       c.index,
		disabled: false,
		client:   client,
	}
	c.index++
	c.edges[edge.id] = edge
	logrus.Infof("Connected with edge %d", edge.id)
	return &pc.AddEdgeResponse{
		Id: int64(edge.id),
	}, nil
}

func (c *Cloud) RemoveEdge(ctx context.Context, request *pc.RemoveEdgeRequest) (*pc.RemoveEdgeResponse, error) {
	id := int(request.Id)
	c.m.Lock()
	defer c.m.Unlock()
	if _, ok := c.edges[id]; !ok {
		return nil, ErrEdgeNotFound
	}
	c.edges[id].disabled = true
	logrus.Infof("Disconnected with source %d", id)
	return &pc.RemoveEdgeResponse{}, nil
}

func (c *Cloud) SendFrame(ctx context.Context, request *pc.EdgeSendFrameRequest) (*pc.EdgeSendFrameResponse, error) {
	id := int(request.Edge)
	c.m.RLock()
	defer c.m.RUnlock()
	if _, ok := c.edges[id]; !ok {
		return nil, ErrEdgeNotFound
	}
	_, err := c.client.SendFrame(context.Background(), &pt.CloudSendFrameRequest{
		Edge:       request.Edge,
		Source:     request.Source,
		Index:      request.Index,
		Dataset:    request.Dataset,
		Content:    request.Content,
		Annotation: request.Annotation,
	})
	if err != nil {
		return nil, err
	}
	return &pc.EdgeSendFrameResponse{}, nil
}

func (c *Cloud) UpdateModel(ctx context.Context, request *pc.TrainerUpdateModelRequest) (*pc.TrainerUpdateModelResponse, error) {
	edge, ok := c.edges[int(request.Edge)]
	if !ok {
		logrus.WithError(ErrEdgeNotFound).Errorf("Update model for edge %d source %d failed", request.Edge, request.Source)
		return &pc.TrainerUpdateModelResponse{}, nil
	}
	_, err := edge.client.UpdateModel(context.Background(), &pe.CloudUpdateModelRequest{
		Source: request.Source,
		Epoch:  request.Epoch,
		Model:  request.Model,
	})
	if err != nil {
		logrus.WithError(err).Errorf("Update model for edge %d source %d failed", request.Edge, request.Source)
		return &pc.TrainerUpdateModelResponse{}, nil
	}
	return &pc.TrainerUpdateModelResponse{}, nil
}

func (c *Cloud) ReportProfile(ctx context.Context, request *pc.TrainerReportProfileRequest) (*pc.TrainerReportProfileResponse, error) {
	edge, ok := c.edges[int(request.Edge)]
	if !ok {
		logrus.WithError(ErrEdgeNotFound).Errorf("Report profile for edge %d source %d failed", request.Edge, request.Source)
		return &pc.TrainerReportProfileResponse{}, nil
	}
	if _, err := edge.client.ReportProfile(context.Background(), &pe.CloudReportProfileRequest{
		Source:   request.Source,
		Begin:    request.Begin,
		Accuracy: request.Accuracy,
	}); err != nil {
		logrus.WithError(err).Errorf("Report profile for edge %d source %d failed", request.Edge, request.Source)
	}
	return &pc.TrainerReportProfileResponse{}, nil
}
