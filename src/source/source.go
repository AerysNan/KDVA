package source

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	pe "vakd/proto/edge"
	"vakd/util"

	"github.com/sirupsen/logrus"
)

type MonitorRecord struct {
	Sent     int
	LastSent int
}

type Source struct {
	ID          int
	Edge        int
	DataDir     string
	Record      MonitorRecord
	Config      util.DatasetInfo
	LastMonitor time.Time

	client pe.EdgeForSourceClient
}

func NewSource(id int, dataset string, dataDir string, client pe.EdgeForSourceClient) (*Source, error) {
	// load dataset information file
	var datasetsInfo util.DatasetsInfo
	file, err := os.Open("cfg_data.json")
	if err != nil {
		return nil, err
	}
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(bytes, &datasetsInfo); err != nil {
		return nil, err
	}
	datasetInfo, ok := datasetsInfo[dataset]
	if !ok {
		return nil, util.ErrDatasetNotFound
	}
	// create source
	s := &Source{
		ID:      id,
		Config:  datasetsInfo[dataset],
		DataDir: dataDir,
		Record: MonitorRecord{
			Sent:     0,
			LastSent: 0,
		},
		client: client,
	}
	// connect source to edge
	for i := 0; i < 5; i++ {
		response, err := client.AddSource(context.Background(), &pe.EdgeAddSourceRequest{
			Source:    int64(id),
			Framerate: int64(datasetInfo.Framerate),
			Dataset:   s.Config.Name,
		})
		if err == nil {
			s.Edge = int(response.Edge)
			logrus.Infof("Connect to edge %d", response.Edge)
			s.LastMonitor = time.Now()
			return s, nil
		}
		time.Sleep(time.Second * 2)
	}
	return nil, err
}

func (s *Source) Start() {
	go s.monitorLoop()
	s.sendFrameLoop()
}

func (s *Source) monitorLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		framerate := float64(s.Record.Sent-s.Record.LastSent) / time.Since(s.LastMonitor).Seconds()
		logrus.Infof("Sent: %d total: %d, FPS: %.3f", s.Record.Sent-s.Record.LastSent, s.Record.Sent, framerate)
		s.LastMonitor = time.Now()
		s.Record.LastSent = s.Record.Sent
		<-timer.C
	}
}

func (s *Source) sendFrameLoop() {
	duration := time.Second / time.Duration(s.Config.Framerate)
	timer := time.NewTimer(duration)
	for {
		<-timer.C
		timer.Reset(duration)
		s.sendFrame(s.Record.Sent)
		s.Record.Sent++
		if s.Record.Sent >= s.Config.Size {
			logrus.Warnf("Stream finished")
			break
		}
	}
}

func (s *Source) sendFrame(index int) {
	file, err := os.Open(filepath.Join(s.DataDir, s.Config.Name, util.SourceGetFrameName(index)))
	if err != nil {
		logrus.WithError(err).Errorf("Failed to open frame %d", index)
		return
	}
	defer file.Close()
	content, err := ioutil.ReadAll(file)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to read frame %d", index)
		return
	}
	_, err = s.client.SendFrame(context.Background(), &pe.SourceSendFrameRequest{
		Source: int64(s.ID),
		Index:  int64(index),
		Blob:   content,
	})
	if err != nil {
		logrus.WithError(err).Errorf("Failed to send frame %d", index)
	}
}
