package cloud

import (
	"context"
	"errors"
	"fmt"
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

func (c *Cloud) UploadFrame(ctx context.Context, request *pc.UploadFrameRequest) (*pc.UploadFrameResponse, error) {
	id := int(request.Edge)
	c.m.RLock()
	defer c.m.RUnlock()
	if _, ok := c.edges[id]; !ok {
		return nil, ErrEdgeNotFound
	}
	if _, err := c.client.AddFrame(context.Background(), &pt.AddFrameRequest{
		Edge:    request.Edge,
		Source:  request.Source,
		Index:   request.Index,
		Content: request.Content,
	}); err != nil {
		return nil, err
	}
	return &pc.UploadFrameResponse{}, nil
}

func (c *Cloud) DeliverModel(ctx context.Context, request *pc.DeliverModelRequest) (*pc.DeliverModelResponse, error) {
	prefix := request.Prefix
	var edgeID, sourceID int
	fmt.Sscanf(prefix, "%d_%d", &edgeID, &sourceID)
	edge, ok := c.edges[edgeID]
	if !ok {
		return nil, ErrEdgeNotFound
	}
	_, err := edge.client.LoadModel(context.Background(), &pe.LoadModelRequest{
		Source: int64(sourceID),
		Epoch:  request.Epoch,
		Model:  request.Model,
	})
	if err != nil {
		return nil, err
	}
	return &pc.DeliverModelResponse{}, nil
}
