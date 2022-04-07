package cloud

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"time"

	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pt "vakd/proto/trainer"
	"vakd/util"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

type Model struct {
	source  int
	edge    int
	version int
}

type Config struct {
	InfFPS int
	RetFPS int
}

type Source struct {
	id            int
	status        int
	version       int
	config        Config
	profile       *pt.RetProfile
	lastRetrained time.Time
}

type Edge struct {
	m        sync.RWMutex
	id       int
	sources  map[int]*Source
	disabled bool
	client   pe.EdgeForCloudClient
}

type Cloud struct {
	pc.CloudForEdgeServer
	m          sync.RWMutex
	retWin     time.Duration
	monitor    *MockNetworkMonitor
	config     util.SimulationConfig
	address    string
	workDir    string
	edges      map[int]*Edge
	client     pt.TrainerForCloudClient
	downloadCh chan *Model
}

func NewCloud(address string, workDir string, config string, client pt.TrainerForCloudClient) (*Cloud, error) {
	cloud := &Cloud{
		m:       sync.RWMutex{},
		workDir: workDir,
		address: address,
		edges:   make(map[int]*Edge),
		retWin:  util.INITIAL_RETRAIN_WINDOW,
		monitor: NewMockNetworkMonitor(util.DOWNLINK_NETWORK_BOTTLENECK),
		client:  client,
	}
	_, err := client.InitTrainer(context.Background(), &pt.InitTrainerRequest{
		WorkDir: workDir,
	})
	if err != nil {
		return nil, err
	}
	file, err := os.Open(config)
	defer func() {
		file.Close()
	}()
	if err != nil {
		return nil, err
	}
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(content, &cloud.config)
	if err != nil {
		return nil, err
	}
	go cloud.downloadLoop()
	go cloud.retrainLoop()
	return cloud, nil
}

func (c *Cloud) downloadLoop() {
	for {
		model := <-c.downloadCh
		edge, ok := c.edges[model.edge]
		if !ok {
			logrus.Errorf("Edge %v not found on model update", model.edge)
			continue
		}
		file, err := os.Open(fmt.Sprintf("%s/%s/%s", c.workDir, util.ModelDir, util.CloudGetModelName(model.edge, model.source, model.version)))
		if err != nil {
			logrus.WithError(err).Errorf("Failed to open model of source %v edge %v v%v", model.source, model.edge, model.version)
			continue
		}
		content, err := ioutil.ReadAll(file)
		if err != nil {
			file.Close()
			logrus.WithError(err).Errorf("Failed to read model of source %v edge %v v%v", model.source, model.edge, model.version)
			continue
		}
		file.Close()
		if _, err = edge.client.UpdateModel(context.Background(), &pe.CloudUpdateModelRequest{
			Source:  int64(model.source),
			Version: int64(model.version),
			Model:   content,
		}); err != nil {
			logrus.WithError(err).Errorf("Failed to update model of source %v edge %v", model.source, model, edge)
			continue
		}
	}
}

func (c *Cloud) retrainLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		c.m.Lock()
		edges := c.edges
		c.m.Unlock()
		for _, edge := range edges {
			if edge.disabled {
				continue
			}
			edge.m.Lock()
			sources := edge.sources
			edge.m.Unlock()
			for _, source := range sources {
				if source.status != util.SOURCE_STATUS_CONNECTED || time.Since(source.lastRetrained) < c.retWin {
					continue
				}
				go func(source *Source, edge *Edge) {
					response, err := c.client.TriggerRetrain(context.Background(), &pt.TriggerRetrainRequest{
						Edge:    int64(edge.id),
						Source:  int64(source.id),
						Version: int64(source.version + 1),
					})
					if err != nil {
						logrus.WithError(err).Errorf("Failed to retrain model of source %v edge %v", source.id, edge.id)
						return
					}
					source.version++
					source.profile = response.RetProfile
					if response.Updated {
						logrus.Infof("Retrain model of source %v edge %v skipped", source.id, edge.id)
						return
					}
					c.downloadCh <- &Model{
						source:  source.id,
						edge:    edge.id,
						version: source.version,
					}
				}(source, edge)
			}
		}
		<-timer.C
	}
}

func (c *Cloud) AddEdge(ctx context.Context, request *pc.AddEdgeRequest) (*pc.AddEdgeResponse, error) {
	connection, err := grpc.Dial(request.Address, grpc.WithInsecure(), grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(1<<30), grpc.MaxCallRecvMsgSize(1<<30)))
	if err != nil {
		logrus.WithError(err).Errorf("Failed to connect to address %v", request.Address)
		return nil, err
	}
	client := pe.NewEdgeForCloudClient(connection)
	c.m.Lock()
	defer c.m.Unlock()
	edge := &Edge{
		m:        sync.RWMutex{},
		id:       int(request.Edge),
		client:   client,
		sources:  make(map[int]*Source),
		disabled: false,
	}
	for _, id := range request.Sources {
		edge.sources[int(id)] = &Source{
			id:      int(id),
			version: 0,
			status:  util.SOURCE_STATUS_CONNECTED,
			profile: &pt.RetProfile{},
			config: Config{
				InfFPS: c.config.EdgeResourceBottleneckFPS / c.config.NSourcesPerEdge,
				RetFPS: c.config.UplinkResourceBottleneckFPS / c.config.NEdges / c.config.NSourcesPerEdge,
			},
			lastRetrained: time.Now(),
		}
	}
	c.edges[edge.id] = edge
	logrus.Infof("Connected with edge %d", edge.id)
	return &pc.AddEdgeResponse{}, nil
}

func (c *Cloud) AddSource(ctx context.Context, request *pc.CloudAddSourceRequest) (*pc.CloudAddSourceResponse, error) {
	c.m.RLock()
	edge, ok := c.edges[int(request.Edge)]
	if !ok {
		c.m.RUnlock()
		return nil, util.ErrEdgeNotFound
	}
	c.m.RUnlock()
	edge.m.Lock()
	_, ok = edge.sources[int(request.Source)]
	if ok {
		edge.m.Unlock()
		return nil, util.ErrSourceExist
	}
	edge.sources[int(request.Source)] = &Source{
		id:      int(request.Source),
		status:  util.SOURCE_STATUS_CONNECTED,
		version: 0,
		profile: &pt.RetProfile{},
		config: Config{
			InfFPS: c.config.EdgeResourceBottleneckFPS / c.config.NSourcesPerEdge,
			RetFPS: c.config.UplinkResourceBottleneckFPS / c.config.NEdges / c.config.NSourcesPerEdge},
		lastRetrained: time.Now(),
	}
	edge.m.Unlock()
	return &pc.CloudAddSourceResponse{}, nil
}

func (c *Cloud) RemoveEdge(ctx context.Context, request *pc.RemoveEdgeRequest) (*pc.RemoveEdgeResponse, error) {
	id := int(request.Edge)
	c.m.Lock()
	defer c.m.Unlock()
	if _, ok := c.edges[id]; !ok {
		return nil, util.ErrEdgeNotFound
	}
	c.edges[id].disabled = true
	logrus.Infof("Disconnected with edge %d", id)
	return &pc.RemoveEdgeResponse{}, nil
}

func (c *Cloud) RemoveSource(ctx context.Context, request *pc.CloudRemoveSourceRequest) (*pc.CloudRemoveSourceResponse, error) {
	c.m.RLock()
	edge, ok := c.edges[int(request.Edge)]
	if !ok {
		c.m.RUnlock()
		return nil, util.ErrEdgeNotFound
	}
	c.m.RUnlock()
	edge.m.Lock()
	source, ok := edge.sources[int(request.Source)]
	if !ok {
		c.m.Unlock()
		return nil, util.ErrSourceNotFound
	}
	source.status = util.SOURCE_STATUS_DISCONNECTED
	c.m.Unlock()
	return &pc.CloudRemoveSourceResponse{}, nil
}

func (c *Cloud) SendFrame(ctx context.Context, request *pc.EdgeSendFrameRequest) (*pc.EdgeSendFrameResponse, error) {
	c.m.RLock()
	if _, ok := c.edges[int(request.Edge)]; !ok {
		c.m.RUnlock()
		return nil, util.ErrEdgeNotFound
	}
	if _, ok := c.edges[int(request.Edge)].sources[int(request.Source)]; !ok {
		c.m.RUnlock()
		return nil, util.ErrSourceNotFound
	}
	c.m.RUnlock()
	path := fmt.Sprintf("%s/%s/%s", c.workDir, util.FrameDir, util.CloudGetFrameName(int(request.Edge), int(request.Source), int(request.Index)))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to open frame %v of edge %v source %v", request.Index, request.Edge, request.Source)
		return &pc.EdgeSendFrameResponse{}, nil
	}
	_, err = file.Write(request.Content)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to write frame %v of edge %v source %v", request.Index, request.Edge, request.Source)
		file.Close()
		return &pc.EdgeSendFrameResponse{}, nil
	}
	file.Close()
	_, err = c.client.SendFrame(context.Background(), &pt.CloudSendFrameRequest{
		Edge:   request.Edge,
		Source: request.Source,
		Index:  request.Index,
	})
	if err != nil {
		return nil, err
	}
	return &pc.EdgeSendFrameResponse{}, nil
}
