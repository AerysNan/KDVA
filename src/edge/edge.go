package edge

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pw "vakd/proto/worker"
	"vakd/util"
)

type Frame struct {
	index  int
	source int
}

type Record struct {
	lastReceivedAcc  int
	receivedAcc      int
	lastProcessedAcc int
	processedAcc     int
	lastUploadedAcc  int
	uploadedAcc      int
}

type Config struct {
	orgFPS    int
	infFPS    int
	retFPS    int
	infSample []int
	retSample []int
}

type Source struct {
	m  sync.RWMutex
	id int

	config Config
	record Record

	status      int
	version     int
	dataset     string
	lastMonitor time.Time
	infQ        chan *Frame
	retQ        chan *Frame
	closeCh     chan struct{}
	updateCh    chan struct{}
}

type Edge struct {
	pe.EdgeForSourceServer
	pe.EdgeForCloudServer
	m            sync.RWMutex
	id           int
	address      string
	workDir      string
	sources      map[int]*Source
	cMonitor     *MockComputationMonitor
	nMonitor     *MockNetworkMonitor
	workerClient pw.WorkerForEdgeClient
	cloudClient  pc.CloudForEdgeClient
}

func NewEdge(id int, address string, workDir string, workerClient pw.WorkerForEdgeClient, cloudClient pc.CloudForEdgeClient) (*Edge, error) {
	edge := &Edge{
		m:            sync.RWMutex{},
		id:           id,
		address:      address,
		sources:      make(map[int]*Source),
		workerClient: workerClient,
		cloudClient:  cloudClient,
		cMonitor:     NewMockComputationMonitor(util.COMPUTATION_BOTTLENECK),
		nMonitor:     NewMockNetworkMonitor(util.UPLINK_NETWORK_BOTTLENECK),
	}
	_, err := cloudClient.AddEdge(context.Background(), &pc.AddEdgeRequest{
		Address: address,
	})
	if err != nil {
		return nil, err
	}
	_, err = workerClient.InitWorker(context.Background(), &pw.InitWorkerRequest{
		WorkDir: edge.workDir,
	})
	if err != nil {
		return nil, err
	}
	edge.workDir = workDir
	go edge.monitorLoop()
	go edge.retrainLoop()
	go edge.inferenceLoop()
	return edge, nil
}

func (e *Edge) monitorLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		func() {
			infT, retT, count := 0.0, 0.0, 0
			ids, delay := make([]int, 0), make(map[int]int)
			e.m.RLock()
			for id, source := range e.sources {
				if source.status != util.SOURCE_STATUS_CLOSED {
					ids = append(ids, id)
				}
			}
			e.m.RUnlock()
			for _, id := range ids {
				source := e.sources[id]
				count++
				infFPS := float64(source.record.processedAcc-source.record.lastProcessedAcc) / time.Since(source.lastMonitor).Seconds()
				retFPS := float64(source.record.uploadedAcc-source.record.lastUploadedAcc) / time.Since(source.lastMonitor).Seconds()
				delay[id] = len(source.infQ)
				source.m.Lock()
				source.record.lastProcessedAcc = source.record.processedAcc
				source.record.lastReceivedAcc = source.record.receivedAcc
				source.record.lastUploadedAcc = source.record.uploadedAcc
				source.m.Unlock()
				infT += infFPS
				retT += retFPS
				source.lastMonitor = time.Now()
			}
			if count > 0 {
				logrus.Infof("S: %d; I: %.2f; U: %.2f; D: %v", count, infT, retT, delay)
			}
		}()
		<-timer.C
	}
}

func (e *Edge) retrainLoop() {
	for {
		ids, weightMap := make([]int, 0), make(map[int]int)
		maxFPS := 0
		e.m.RLock()
		for id, source := range e.sources {
			if len(source.infQ) == 0 && source.status == util.SOURCE_STATUS_DISCONNECTED {
				source.status = util.SOURCE_STATUS_CLOSED
				source.closeCh <- struct{}{}
			}
			if source.status == util.SOURCE_STATUS_CLOSED {
				continue
			}
			ids = append(ids, id)
			source.m.RLock()
			if source.config.retFPS > maxFPS {
				maxFPS = source.config.retFPS
			}
			weightMap[id] = source.config.retFPS
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
				source := e.sources[id]
				if weightMap[id] < w || len(source.retQ) == 0 {
					continue
				}
				timer := time.NewTimer(util.TIMEOUT_DURATION)
				select {
				case frame := <-source.retQ:
					logrus.Debugf("Upload frame %d of source %d", frame.index, frame.source)
					path := fmt.Sprintf("%s/%s/%s", e.workDir, util.FrameDir, util.EdgeGetFrameName(int(frame.source), int(frame.index)))
					file, err := os.Open(path)
					if err != nil {
						logrus.WithError(err).Errorf("Failed to open frame %d of source %d", frame.index, frame.source)
						continue
					}
					content, err := ioutil.ReadAll(file)
					if err != nil {
						logrus.WithError(err).Errorf("Failed to read frame %d of source %d", frame.index, frame.source)
						file.Close()
						continue
					}
					file.Close()
					_, err = e.cloudClient.SendFrame(context.Background(), &pc.EdgeSendFrameRequest{
						Edge:    int64(e.id),
						Source:  int64(frame.source),
						Index:   int64(frame.index),
						Content: content,
					})
					if err != nil {
						logrus.WithError(err).Errorf("Failed to upload frame %d of source %d", frame.index, frame.source)
					}
					source.record.uploadedAcc++
				case <-timer.C:
					continue
				}
			}
		}
	}
}

func (e *Edge) inferenceLoop() {
	for {
		ids, weightMap := make([]int, 0), make(map[int]int)
		maxFPS := 0
		e.m.RLock()
		for id, source := range e.sources {
			if len(source.infQ) == 0 && source.status == util.SOURCE_STATUS_DISCONNECTED {
				source.status = util.SOURCE_STATUS_CLOSED
				source.closeCh <- struct{}{}
			}
			if source.status == util.SOURCE_STATUS_CLOSED {
				continue
			}
			ids = append(ids, id)
			source.m.RLock()
			if source.config.infFPS > maxFPS {
				maxFPS = source.config.infFPS
			}
			weightMap[id] = source.config.infFPS
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
				source := e.sources[id]
				if weightMap[id] < w || len(source.infQ) == 0 {
					continue
				}
				timer := time.NewTimer(util.TIMEOUT_DURATION)
				select {
				case frame := <-source.infQ:
					logrus.Debugf("Infer frame %d of source %d", frame.index, frame.source)
					_, err := e.workerClient.InferFrame(context.Background(), &pw.InferFrameRequest{
						Source: int64(frame.source),
						Index:  int64(frame.index),
					})
					if err != nil {
						logrus.WithError(err).Errorf("Infer frame %d of source %d failed", frame.index, frame.source)
						return
					}
					source.record.processedAcc++
				case <-timer.C:
					continue
				}
			}
		}
	}
}

func (e *Edge) AddSource(ctx context.Context, request *pe.EdgeAddSourceRequest) (*pe.EdgeAddSourceResponse, error) {
	source := &Source{
		m:  sync.RWMutex{},
		id: int(request.Source),
		record: Record{
			lastReceivedAcc:  0,
			receivedAcc:      0,
			lastProcessedAcc: 0,
			processedAcc:     0,
			lastUploadedAcc:  0,
			uploadedAcc:      0,
		},
		status:   util.SOURCE_STATUS_CONNECTED,
		version:  0,
		dataset:  request.Dataset,
		infQ:     make(chan *Frame, 100),
		retQ:     make(chan *Frame, 100),
		updateCh: make(chan struct{}),
		closeCh:  make(chan struct{}),
	}
	if _, err := e.workerClient.UpdateModel(context.Background(), &pw.EdgeUpdateModelRequest{
		Source:  int64(source.id),
		Version: 0,
	}); err != nil {
		logrus.WithError(err).Errorf("Failed to add model for source %d", source.id)
		return nil, util.ErrAddModel
	}
	logrus.Debugf("Model added for source %d", source.id)
	if _, err := e.cloudClient.AddSource(context.Background(), &pc.CloudAddSourceRequest{
		Edge:   int64(e.id),
		Source: int64(source.id),
	}); err != nil {
		logrus.WithError(err).Errorf("Failed to add source %d to cloud", e.id)
		return nil, util.ErrAddSource
	}
	cResource, nResource := e.cMonitor.GetResource(), e.nMonitor.GetResource()
	count := 0
	e.m.RLock()
	for _, source := range e.sources {
		if source.status == util.SOURCE_STATUS_CONNECTED {
			count++
		}
	}
	source.config.infFPS = int(cResource) / (count + 1)
	source.config.retFPS = int(nResource) / (count + 1)
	for _, source := range e.sources {
		source.m.Lock()
		if source.status != util.SOURCE_STATUS_CONNECTED {
			source.m.Unlock()
			continue
		}
		source.config.infFPS = int(cResource) * count / (count + 1)
		source.config.infSample = util.GenerateSamplePosition(source.config.infFPS, source.config.orgFPS, 0)
		source.config.retFPS = int(nResource) * count / (count + 1)
		source.config.retSample = util.GenerateSamplePosition(source.config.retFPS, source.config.orgFPS, 0)
		source.m.Unlock()
	}
	e.m.RUnlock()
	e.m.Lock()
	e.sources[source.id] = source
	e.m.Unlock()
	logrus.Infof("Connect source %d", source.id)
	source.lastMonitor = time.Now()
	return &pe.EdgeAddSourceResponse{
		Edge: int64(e.id),
	}, nil
}

func (e *Edge) RemoveSource(ctx context.Context, request *pe.RemoveSourceRequest) (*pe.RemoveSourceResponse, error) {
	id := int(request.Source)
	source, ok := e.sources[id]
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.status = util.SOURCE_STATUS_DISCONNECTED
	logrus.Infof("Disconnect with source %d", id)
	if _, err := e.cloudClient.RemoveSource(context.Background(), &pc.CloudRemoveSourceRequest{
		Edge:   int64(e.id),
		Source: request.Source,
	}); err != nil {
		logrus.WithError(err).Errorf("Failed to remove source %v from cloud", request.Source)
	}
	<-source.closeCh
	logrus.Infof("Close source %d", id)
	return &pe.RemoveSourceResponse{}, nil
}

func (e *Edge) SendFrame(ctx context.Context, request *pe.SourceSendFrameRequest) (*pe.SourceSendFrameResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	source, ok := e.sources[id]
	e.m.RUnlock()
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.m.RLock()
	needInf, needRet := util.Exist(source.config.infSample, int(request.Index)%source.config.orgFPS), util.Exist(source.config.retSample, int(request.Index)%source.config.orgFPS)
	source.m.Unlock()
	if needInf || needRet {
		path := fmt.Sprintf("%s/%s/%s", e.workDir, util.FrameDir, util.EdgeGetFrameName(int(request.Source), int(request.Index)))
		file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
		if err != nil {
			logrus.WithError(err).Errorf("Open frame %d for source %d failed", request.Index, request.Source)
			return &pe.SourceSendFrameResponse{}, nil
		}
		_, err = file.Write(request.Content)
		if err != nil {
			logrus.WithError(err).Errorf("Write frame %d for source %d failed", request.Index, request.Source)
			file.Close()
			return &pe.SourceSendFrameResponse{}, nil
		}
		file.Close()
	}
	frame := &Frame{
		index:  int(request.Index),
		source: id,
	}
	if needInf {
		source.infQ <- frame
	}
	if needRet {
		source.retQ <- frame
	}
	logrus.Debugf("Receive frame %d from source %d", request.Index, id)
	source.m.Lock()
	source.record.receivedAcc++
	source.m.Unlock()
	return &pe.SourceSendFrameResponse{}, nil
}

func (e *Edge) UpdateModel(ctx context.Context, request *pe.CloudUpdateModelRequest) (*pe.CloudUpdateModelResponse, error) {
	id := int(request.Source)
	e.m.RLock()
	source, ok := e.sources[id]
	e.m.RUnlock()
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.version++
	path := fmt.Sprintf("%s/%s/%s", e.workDir, util.ModelDir, util.EdgeGetModelName(int(request.Source), source.version))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to open model for source %d v%d", request.Source, source.version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	defer file.Close()
	_, err = file.Write(request.Model)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to write model for source %d v%d", request.Source, source.version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	_, err = e.workerClient.UpdateModel(context.Background(), &pw.EdgeUpdateModelRequest{
		Source:  request.Source,
		Version: int64(source.version),
	})
	if err != nil {
		logrus.WithError(err).Errorf("Failed to update model for source %d v%d", request.Source, source.version)
		return &pe.CloudUpdateModelResponse{}, nil
	}
	logrus.Infof("Update model for source %d to v%d", source.id, source.version)
	// source.updateCh <- struct{}{}
	return &pe.CloudUpdateModelResponse{}, nil
}

func (e *Edge) UpdateConfig(ctx context.Context, request *pe.CloudUpdateConfigRequest) (*pe.CloudUpdateConfigResponse, error) {
	source, ok := e.sources[int(request.Source)]
	if !ok {
		return nil, util.ErrSourceNotFound
	}
	source.m.Lock()
	source.config.infFPS = int(request.InfFps)
	source.config.retFPS = int(request.RetFps)
	source.config.infSample = util.GenerateSamplePosition(source.config.infFPS, source.config.orgFPS, 0)
	source.config.retSample = util.GenerateSamplePosition(source.config.retFPS, source.config.orgFPS, 0)
	source.m.Unlock()
	return &pe.CloudUpdateConfigResponse{}, nil
}
