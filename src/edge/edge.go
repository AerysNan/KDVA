package edge

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pw "vakd/proto/worker"
	"vakd/util"

	"github.com/sirupsen/logrus"
)

type Frame struct {
	Index  int
	Source int
}

type MonitorRecord struct {
	LastReceived  int
	Received      int
	LastProcessed int
	Processed     int
	LastUploaded  int
	Uploaded      int
}

type Source struct {
	ID int

	Config util.SourceConfig
	record MonitorRecord

	Version     int
	Dataset     string
	LastMonitor time.Time

	infQ     chan *Frame
	retQ     chan *Frame
	closeCh  chan struct{}
	updateCh chan struct{}
}

type Edge struct {
	pe.EdgeForSourceServer
	pe.EdgeForCloudServer
	ID      int
	Address string
	WorkDir string
	Config  util.SimulationConfig
	Sources map[int]*Source

	m            sync.RWMutex
	workerClient pw.WorkerForEdgeClient
	cloudClient  pc.CloudForEdgeClient
}

func NewEdge(id int, address string, workDir string, config string, workerClient pw.WorkerForEdgeClient, cloudClient pc.CloudForEdgeClient) (*Edge, error) {
	// create edge
	e := &Edge{
		m:            sync.RWMutex{},
		ID:           id,
		Address:      address,
		Sources:      make(map[int]*Source),
		workerClient: workerClient,
		cloudClient:  cloudClient,
	}
	// read simulation config
	file, err := os.Open(config)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(bytes, &e.Config); err != nil {
		return nil, err
	}
	// connect to worker
	e.WorkDir = workDir
	_, err = workerClient.InitWorker(context.Background(), &pw.InitWorkerRequest{
		WorkDir: e.WorkDir,
	})
	if err != nil {
		return nil, err
	}
	return e, nil
}

func (e *Edge) Start() {
	// connect to cloud
	sources := make([]*pc.SourceInfo, 0)
	for id, source := range e.Sources {
		sources = append(sources, &pc.SourceInfo{
			Id:                int64(id),
			OriginalFramerate: int64(source.Config.OriginalFramerate),
		})
	}
	if _, err := e.cloudClient.AddEdge(context.Background(), &pc.AddEdgeRequest{
		Address: e.Address,
		Edge:    int64(e.ID),
		Sources: sources,
	}); err != nil {
		logrus.WithError(err).Error("Failed to connect to cloud")
	} else {
		logrus.Info("Edge started")
	}
	for _, s := range e.Sources {
		s.LastMonitor = time.Now()
	}
	go e.monitorLoop()
	go e.retrainLoop()
	go e.inferenceLoop()
}

func (e *Edge) monitorLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		func() {
			totalInferenceFPS, totalUploadFPS, count := 0.0, 0.0, 0
			for _, source := range e.Sources {
				count++
				infFPS := float64(source.record.Processed-source.record.LastProcessed) / time.Since(source.LastMonitor).Seconds()
				retFPS := float64(source.record.Uploaded-source.record.LastUploaded) / time.Since(source.LastMonitor).Seconds()
				source.record.LastProcessed = source.record.Processed
				source.record.LastReceived = source.record.Received
				source.record.LastUploaded = source.record.Uploaded
				totalInferenceFPS += infFPS
				totalUploadFPS += retFPS
				source.LastMonitor = time.Now()
			}
			if count > 0 {
				logrus.Infof("Sources: %d, inference: %.2f FPS, upload: %.2f FPS", count, totalInferenceFPS, totalUploadFPS)
			}
		}()
		<-timer.C
	}
}

func (e *Edge) retrainLoop() {
	for {
		weightMap, weightList := make(map[int]int), make([]int, 0)
		maxFPS := 0
		e.m.RLock()
		for id, source := range e.Sources {
			if source.Config.UploadingFramerate > maxFPS {
				maxFPS = source.Config.UploadingFramerate
			}
			weightMap[id] = source.Config.UploadingFramerate
			weightList = append(weightList, source.Config.UploadingFramerate)
		}
		e.m.RUnlock()
		gcd := util.GCDList(weightList)
		for w := 1; w <= maxFPS/gcd; w++ {
			for id, s := range e.Sources {
				if weightMap[id] < w*gcd || len(s.retQ) == 0 {
					continue
				}
				timer := time.NewTimer(util.QUEUE_TIMEOUT)
				select {
				case frame := <-s.retQ:
					logrus.Debugf("Upload frame %d of source %d", frame.Index, frame.Source)
					path := filepath.Join(e.WorkDir, util.FrameDir, util.EdgeGetFrameName(int(frame.Source), int(frame.Index)))
					file, err := os.Open(path)
					if err != nil {
						logrus.WithError(err).Errorf("Failed to open frame %d of source %d", frame.Index, frame.Source)
						continue
					}
					content, err := ioutil.ReadAll(file)
					if err != nil {
						logrus.WithError(err).Errorf("Failed to read frame %d of source %d", frame.Index, frame.Source)
						file.Close()
						continue
					}
					file.Close()
					_, err = e.cloudClient.SendFrame(context.Background(), &pc.EdgeSendFrameRequest{
						Edge:    int64(e.ID),
						Source:  int64(frame.Source),
						Index:   int64(frame.Index),
						Content: content,
					})
					if err != nil {
						logrus.WithError(err).Errorf("Failed to upload frame %d of source %d", frame.Index, frame.Source)
					}
					s.record.Uploaded++
				case <-timer.C:
					continue
				}
			}
		}
	}
}

func (e *Edge) inferenceLoop() {
	for {
		weightMap, weightList := make(map[int]int), make([]int, 0)
		maxFPS := 0
		e.m.RLock()
		for id, source := range e.Sources {
			if source.Config.InferenceFramerate > maxFPS {
				maxFPS = source.Config.InferenceFramerate
			}
			weightMap[id] = source.Config.InferenceFramerate
			weightList = append(weightList, source.Config.InferenceFramerate)
		}
		e.m.RUnlock()
		gcd := util.GCDList(weightList)
		for w := 1; w <= maxFPS/gcd; w++ {
			for id, s := range e.Sources {
				if weightMap[id] < w || len(s.infQ) == 0 {
					continue
				}
				timer := time.NewTimer(util.QUEUE_TIMEOUT)
				select {
				case frame := <-s.infQ:
					logrus.Debugf("Infer frame %d of source %d", frame.Index, frame.Source)
					_, err := e.workerClient.InferFrame(context.Background(), &pw.InferFrameRequest{
						Source: int64(frame.Source),
						Index:  int64(frame.Index),
					})
					if err != nil {
						logrus.WithError(err).Errorf("Infer frame %d of source %d failed", frame.Index, frame.Source)
						return
					}
					s.record.Processed++
				case <-timer.C:
					continue
				}
			}
		}
	}
}

func (e *Edge) AddSource(ctx context.Context, request *pe.EdgeAddSourceRequest) (*pe.EdgeAddSourceResponse, error) {
	s := &Source{
		ID: int(request.Source),
		record: MonitorRecord{
			LastReceived:  0,
			Received:      0,
			LastProcessed: 0,
			Processed:     0,
			LastUploaded:  0,
			Uploaded:      0,
		},
		Version:  0,
		Dataset:  request.Dataset,
		infQ:     make(chan *Frame, 100),
		retQ:     make(chan *Frame, 100),
		updateCh: make(chan struct{}),
		closeCh:  make(chan struct{}),
	}
	if _, err := e.workerClient.UpdateModel(context.Background(), &pw.EdgeUpdateModelRequest{
		Source:  int64(s.ID),
		Version: 0,
	}); err != nil {
		logrus.WithError(err).Errorf("Failed to add model for source %d", s.ID)
		return nil, util.ErrAddModel
	}
	logrus.Debugf("Model added for source %d", s.ID)
	cResource, nResource := e.Config.EdgeFPS, e.Config.UplinkFPS
	s.Config.OriginalFramerate = int(request.Framerate)
	s.Config.InferenceFramerate = int(cResource) / e.Config.NSourcesPerEdge
	s.Config.UploadingFramerate = int(nResource) / e.Config.NSourcesPerEdge
	s.Config.InfSamplePos = util.GenerateSamplePosition(s.Config.InferenceFramerate, s.Config.OriginalFramerate, 0)
	s.Config.RetSamplePos = util.GenerateSamplePosition(s.Config.UploadingFramerate, s.Config.OriginalFramerate, 0)
	e.Sources[s.ID] = s
	logrus.Infof("Connect source %d", s.ID)
	if len(e.Sources) == e.Config.NSourcesPerEdge {
		defer e.Start()
	}
	return &pe.EdgeAddSourceResponse{
		Edge: int64(e.ID),
	}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SourceSendFrameRequest) (*pe.SourceSendFrameResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	source, ok := e.Sources[id]
	e.m.RUnlock()
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	needInf, needRet := util.Exist(source.Config.InfSamplePos, int(request.Index)%source.Config.OriginalFramerate), util.Exist(source.Config.RetSamplePos, int(request.Index)%source.Config.OriginalFramerate)
	if needInf || needRet {
		path := filepath.Join(e.WorkDir, util.FrameDir, util.EdgeGetFrameName(int(request.Source), int(request.Index)))
		file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
		if err != nil {
			logrus.WithError(err).Errorf("Open frame %d for source %d failed", request.Index, request.Source)
			return &pe.SourceSendFrameResponse{}, nil
		}
		_, err = file.Write(request.Blob)
		if err != nil {
			logrus.WithError(err).Errorf("Write frame %d for source %d failed", request.Index, request.Source)
			file.Close()
			return &pe.SourceSendFrameResponse{}, nil
		}
		file.Close()
	}
	frame := &Frame{
		Index:  int(request.Index),
		Source: id,
	}
	if needInf {
		source.infQ <- frame
	}
	if needRet {
		source.retQ <- frame
	}
	logrus.Debugf("Receive frame %d from source %d", request.Index, id)
	source.record.Received++
	return &pe.SourceSendFrameResponse{}, nil
}

func (e *Edge) UpdateModel(ctx context.Context, request *pe.CloudUpdateModelRequest) (*pe.CloudUpdateModelResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	source, ok := e.Sources[id]
	e.m.RUnlock()
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.Version++
	path := filepath.Join(e.WorkDir, util.ModelDir, util.EdgeGetModelName(int(request.Source), source.Version))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to open model for source %d v%d", request.Source, source.Version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	defer file.Close()
	_, err = file.Write(request.Model)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to write model for source %d v%d", request.Source, source.Version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	_, err = e.workerClient.UpdateModel(context.Background(), &pw.EdgeUpdateModelRequest{
		Source:  request.Source,
		Version: int64(source.Version),
	})
	if err != nil {
		logrus.WithError(err).Errorf("Failed to update model for source %d v%d", request.Source, source.Version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	logrus.Infof("Update model for source %d to v%d", source.ID, source.Version)
	// source.updateCh <- struct{}{}
	return &pe.CloudUpdateModelResponse{}, nil
}

func (e *Edge) UpdateConfig(ctx context.Context, request *pe.CloudUpdateConfigRequest) (*pe.CloudUpdateConfigResponse, error) {
	source, ok := e.Sources[int(request.Source)]
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.Config.InferenceFramerate = int(request.InfFramerate)
	source.Config.UploadingFramerate = int(request.RetFramerate)
	source.Config.InfSamplePos = util.GenerateSamplePosition(source.Config.InferenceFramerate, source.Config.OriginalFramerate, 0)
	source.Config.RetSamplePos = util.GenerateSamplePosition(source.Config.UploadingFramerate, source.Config.OriginalFramerate, 0)
	logrus.Debugf("Source %v update config inf: %d FPS, ret: %d FPS", source.ID, source.Config.InferenceFramerate, source.Config.UploadingFramerate)
	return &pe.CloudUpdateConfigResponse{}, nil
}
