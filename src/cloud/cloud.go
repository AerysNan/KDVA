package cloud

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pt "vakd/proto/trainer"
	"vakd/util"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

type Model struct {
	Source  int
	Edge    int
	Version int
}

type Source struct {
	ID            int
	Version       int
	Current       int
	LastRetrained int
	Config        util.SourceConfig

	profile []float64
}

type Edge struct {
	ID      int
	Sources map[int]*Source

	client pe.EdgeForCloudClient
}

type Cloud struct {
	pc.CloudForEdgeServer

	Address string
	WorkDir string
	Edges   map[int]*Edge
	Config  util.SimulationConfig

	client     pt.TrainerForCloudClient
	downloadCh chan *Model
}

func NewCloud(address string, workDir string, config string, client pt.TrainerForCloudClient) (*Cloud, error) {
	// create c
	c := &Cloud{
		WorkDir: workDir,
		Address: address,
		Edges:   make(map[int]*Edge),
		client:  client,
	}
	// read simulation config
	file, err := os.Open(config)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(content, &c.Config)
	if err != nil {
		return nil, err
	}
	//connect to trainer
	_, err = client.InitTrainer(context.Background(), &pt.InitTrainerRequest{
		WorkDir: workDir,
	})
	if err != nil {
		return nil, err
	}
	return c, nil
}

func (c *Cloud) Start() {
	logrus.Info("Cloud started")
	go c.downloadLoop()
	go c.retrainLoop()
}

func (c *Cloud) downloadLoop() {
	for {
		model := <-c.downloadCh
		edge, ok := c.Edges[model.Edge]
		if !ok {
			logrus.Errorf("Edge %v not found on model update", model.Edge)
			continue
		}
		file, err := os.Open(filepath.Join(c.WorkDir, util.ModelDir, util.CloudGetModelName(model.Edge, model.Source, model.Version)))
		if err != nil {
			logrus.WithError(err).Errorf("Failed to open model of source %v edge %v version %v", model.Source, model.Edge, model.Version)
			continue
		}
		content, err := ioutil.ReadAll(file)
		if err != nil {
			file.Close()
			logrus.WithError(err).Errorf("Failed to read model of source %v edge %v version %v", model.Source, model.Edge, model.Version)
			continue
		}
		file.Close()
		if _, err = edge.client.UpdateModel(context.Background(), &pe.CloudUpdateModelRequest{
			Source:  int64(model.Source),
			Version: int64(model.Version),
			Model:   content,
		}); err != nil {
			logrus.WithError(err).Errorf("Failed to update model of source %v edge %v", model.Source, model, edge)
			continue
		} else {
			c.reallocateResource()
		}
	}
}

func (c *Cloud) reallocateResource() {
	for _, e := range c.Edges {
		for _, s := range e.Sources {
			s.Config.InferenceFramerate = c.Config.EdgeFPS / c.Config.NSourcesPerEdge
			s.Config.UploadingFramerate = c.Config.EdgeFPS / c.Config.NEdges / c.Config.NSourcesPerEdge
			go func(source *Source, edge *Edge) {
				if _, err := edge.client.UpdateConfig(context.Background(), &pe.CloudUpdateConfigRequest{
					InfFramerate: int64(source.Config.InferenceFramerate),
					RetFramerate: int64(source.Config.UploadingFramerate),
				}); err != nil {
					logrus.WithError(err).Errorf("Update config of source %v edge %v failed", source.ID, edge.ID)
				}
			}(s, e)
		}
	}
}

func (c *Cloud) retrainLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		for _, e := range c.Edges {
			for _, s := range e.Sources {
				if s.Current-s.LastRetrained < c.Config.RetrainWindow {
					continue
				}
				s.LastRetrained = s.Current
				go func(source *Source, edge *Edge) {
					response, err := c.client.TriggerRetrain(context.Background(), &pt.TriggerRetrainRequest{
						Edge:    int64(edge.ID),
						Source:  int64(source.ID),
						Version: int64(source.Version + 1),
					})
					if err != nil {
						logrus.WithError(err).Errorf("Failed to retrain model of source %v edge %v", source.ID, edge.ID)
						return
					}
					source.Version++
					source.profile = response.Profile
					if response.Updated {
						logrus.Infof("Retrain model of source %v edge %v skipped", source.ID, edge.ID)
						return
					}
					c.downloadCh <- &Model{
						Source:  source.ID,
						Edge:    edge.ID,
						Version: source.Version,
					}
				}(s, e)
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
	edge := &Edge{
		ID:      int(request.Edge),
		client:  client,
		Sources: make(map[int]*Source),
	}
	c.Edges[edge.ID] = edge
	for _, s := range request.Sources {
		edge.Sources[int(s.Id)] = &Source{
			ID:      int(s.Id),
			Version: 0,
			Current: 0,
			profile: nil,
			Config: util.SourceConfig{
				OriginalFramerate:  int(s.OriginalFramerate),
				InferenceFramerate: c.Config.EdgeFPS / c.Config.NSourcesPerEdge,
				UploadingFramerate: c.Config.UplinkFPS / c.Config.NSourcesPerEdge / c.Config.NEdges,
			},
			LastRetrained: -1,
		}
	}
	logrus.Infof("Connected with edge %d", edge.ID)
	if len(c.Edges) == c.Config.NEdges {
		defer c.Start()
	}
	return &pc.AddEdgeResponse{}, nil
}

func (c *Cloud) SendFrame(ctx context.Context, request *pc.EdgeSendFrameRequest) (*pc.EdgeSendFrameResponse, error) {
	if _, ok := c.Edges[int(request.Edge)]; !ok {
		return nil, util.ErrEdgeNotFound
	}
	s, ok := c.Edges[int(request.Edge)].Sources[int(request.Source)]
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	path := filepath.Join(c.WorkDir, util.FrameDir, util.CloudGetFrameName(int(request.Edge), int(request.Source), int(request.Index)))
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
	s.Current = int(request.Index)
	return &pc.EdgeSendFrameResponse{}, nil
}
