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
	ps "vakd/proto/source"

	"github.com/sirupsen/logrus"
)

const (
	MONITOR_INTERVAL  = time.Millisecond * 1000
	INITIAL_FRAMERATE = 2
)

type Source struct {
	ps.SourceForEdgeServer
	id           int
	edge         int
	currentIndex int
	originalFPS  int
	currentFPS   int
	count        int
	sent         int
	eval         bool
	active       bool
	lastMonitor  time.Time
	dataset      string
	address      string
	client       pe.EdgeForSourceClient
	m            sync.RWMutex
}

func NewSource(dataset string, address string, fps int, eval bool, client pe.EdgeForSourceClient) (*Source, error) {
	source := &Source{
		m:            sync.RWMutex{},
		currentIndex: 0,
		sent:         0,
		eval:         eval,
		active:       true,
		originalFPS:  fps,
		currentFPS:   INITIAL_FRAMERATE,
		dataset:      dataset,
		address:      address,
		client:       client,
	}
	files, err := ioutil.ReadDir(fmt.Sprintf("data/%s", dataset))
	if err != nil {
		return nil, err
	}
	source.count = 0
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".jpg" {
			source.count++
		}
	}
	response, err := client.AddSource(context.Background(), &pe.AddSourceRequest{
		Address: address,
		Fps:     int64(INITIAL_FRAMERATE),
		Dataset: source.dataset,
	})
	if err != nil {
		return nil, err
	}
	source.id = int(response.Id)
	source.edge = int(response.Edge)
	source.lastMonitor = time.Now()
	go source.sendFrameLoop()
	go source.monitorLoop()
	return source, nil
}

func (s *Source) monitorLoop() {
	timer := time.NewTicker(MONITOR_INTERVAL)
	for {
		if !s.active {
			return
		}
		fps := float64(s.sent) / time.Since(s.lastMonitor).Seconds()
		logrus.Infof("Current frame: %d; Config FPS: %d fps; Actual FPS %.3f fps", s.currentIndex, s.currentFPS, fps)
		s.lastMonitor = time.Now()
		s.m.Lock()
		s.sent = 0
		s.m.Unlock()
		<-timer.C
	}
}

func (s *Source) sendFrameLoop() {
	s.m.RLock()
	duration := time.Second / time.Duration(s.currentFPS)
	s.m.RUnlock()
	timer := time.NewTimer(duration)
	for {
		s.m.RLock()
		currentFPS := s.currentFPS
		s.m.RUnlock()
		strideList := make([]int, currentFPS)
		remainder := s.originalFPS
		for i := 0; i < currentFPS; i++ {
			strideList[i] = s.originalFPS / currentFPS
			remainder -= strideList[i]
		}
		for i, index := 0, 0; i < remainder; i++ {
			strideList[index]++
			index += currentFPS / remainder
		}
		duration = time.Second / time.Duration(currentFPS)
		for i := 0; i < currentFPS; i++ {
			<-timer.C
			timer.Reset(duration)
			s.sendFrame(s.currentIndex)
			s.currentIndex += strideList[i]
			if s.currentIndex >= s.count {
				logrus.Warnf("Stream %d exited since all frames are sent to edge", s.id)
				if _, err := s.client.RemoveSource(context.Background(), &pe.RemoveSourceRequest{
					Id: int64(s.id),
				}); err != nil {
					logrus.WithError(err).Error("Failed to disconnect from edge")
				}
				s.active = false
				if s.eval {
					output, err := exec.Command("tools/evaluate_system.py", "--id", fmt.Sprintf("%v_%v", s.edge, s.id), "--dataset", s.dataset).Output()
					if err != nil {
						logrus.WithError(err).Error("Execute evaluation failed")
					} else {
						logrus.Infof("Evaluation result: %s", output)
					}
				}
				os.Exit(0)
			}
		}
	}
}

func (s *Source) sendFrame(current int) {
	file, err := os.Open(fmt.Sprintf("data/%s/%06d.jpg", s.dataset, current))
	if err != nil {
		logrus.WithError(err).Errorf("Open frame %d failed", current)
		return
	}
	defer file.Close()
	content, err := ioutil.ReadAll(file)
	if err != nil {
		logrus.WithError(err).Errorf("Read frame %d failed", current)
		return
	}
	_, err = s.client.SendFrame(context.Background(), &pe.SourceSendFrameRequest{
		Source:  int64(s.id),
		Index:   int64(current),
		Content: content,
	})
	if err != nil {
		logrus.WithError(err).Errorf("Send frame %d failed", current)
	}
	s.m.Lock()
	s.sent++
	s.m.Unlock()
}

func (s *Source) SetFramerate(ctx context.Context, request *ps.SetFramerateRequest) (*ps.SetFramerateResponse, error) {
	s.m.Lock()
	s.currentFPS = int(request.FrameRate)
	s.m.Unlock()
	return &ps.SetFramerateResponse{}, nil
}
