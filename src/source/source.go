package source

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
	pe "vakd/proto/edge"
	"vakd/util"

	"github.com/sirupsen/logrus"
)

type Record struct {
	sentAcc     int
	lastSentAcc int
}

type Source struct {
	id          int
	fps         int
	edge        int
	count       int
	eval        bool
	dataDir     string
	dataset     string
	record      Record
	lastMonitor time.Time
	client      pe.EdgeForSourceClient
	m           sync.RWMutex
}

func NewSource(id int, dataset string, dataDir string, fps int, eval bool, client pe.EdgeForSourceClient) (*Source, error) {
	source := &Source{
		m:  sync.RWMutex{},
		id: id,
		record: Record{
			sentAcc:     0,
			lastSentAcc: 0,
		},
		eval:    eval,
		fps:     fps,
		dataDir: dataDir,
		dataset: dataset,
		client:  client,
	}
	files, err := ioutil.ReadDir(fmt.Sprintf("%s/%s", dataDir, dataset))
	if err != nil {
		return nil, err
	}
	source.count = 0
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".jpg" {
			source.count++
		}
	}
	response, err := client.AddSource(context.Background(), &pe.EdgeAddSourceRequest{
		Source:  int64(id),
		Fps:     int64(fps),
		Dataset: source.dataset,
	})
	if err != nil {
		return nil, err
	}
	logrus.Infof("Assign source ID %d", source.id)
	source.edge = int(response.Edge)
	source.lastMonitor = time.Now()

	return source, nil
}

func (s *Source) Start() {
	go s.monitorLoop()
	s.sendFrameLoop()
}

func (s *Source) monitorLoop() {
	timer := time.NewTicker(util.MONITOR_INTERVAL)
	for {
		fps := float64(s.record.sentAcc-s.record.lastSentAcc) / time.Since(s.lastMonitor).Seconds()
		logrus.Infof("Sent %d frames %.3f FPS", s.record.sentAcc, fps)
		s.lastMonitor = time.Now()
		s.m.Lock()
		s.record.lastSentAcc = s.record.sentAcc
		s.m.Unlock()
		<-timer.C
	}
}

func (s *Source) sendFrameLoop() {
	s.m.RLock()
	duration := time.Second / time.Duration(s.fps)
	s.m.RUnlock()
	timer := time.NewTimer(duration)
	for {
		<-timer.C
		timer.Reset(duration)
		s.sendFrame(s.record.sentAcc)
		s.record.sentAcc++
		if s.record.sentAcc >= s.count {
			logrus.Warnf("Source %d finished", s.id)
			if _, err := s.client.RemoveSource(context.Background(), &pe.RemoveSourceRequest{
				Source: int64(s.id),
			}); err != nil {
				logrus.WithError(err).Error("Failed to disconnect from edge")
			}
			if s.eval {
				output, err := exec.Command("tools/range_eval.py", "-r", fmt.Sprintf("%v_%v", s.edge, s.id), "-d", s.dataset, "-n", "12").Output()
				if err != nil {
					logrus.WithError(err).Error("Failed to execute evaluation")
				} else {
					logrus.Infof("Evaluation result:\n%s", output)
				}
			}
			break
		}
	}
}

func (s *Source) sendFrame(index int) {
	file, err := os.Open(fmt.Sprintf("%s/%s/%s", s.dataDir, s.dataset, util.SourceGetFrameName(index)))
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
		Source:  int64(s.id),
		Index:   int64(index),
		Content: content,
	})
	if err != nil {
		logrus.WithError(err).Errorf("Failed to send frame %d", index)
	}
}
