package source

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"
	pe "vakd/proto/edge"
	ps "vakd/proto/source"

	"github.com/sirupsen/logrus"
)

const (
	MONITOR_INTERVAL = time.Millisecond * 1000
)

type Source struct {
	ps.SourceForEdgeServer
	id           int
	currentIndex int
	originalFPS  int
	currentFPS   int
	count        int
	sent         int
	start        time.Time
	datadir      string
	address      string
	client       pe.EdgeForSourceClient
	m            sync.RWMutex
}

func NewSource(path string, address string, fps int, client pe.EdgeForSourceClient) (*Source, error) {
	source := &Source{
		m:            sync.RWMutex{},
		currentIndex: 0,
		sent:         0,
		originalFPS:  fps,
		currentFPS:   fps,
		datadir:      path,
		address:      address,
		client:       client,
	}
	files, err := ioutil.ReadDir(path)
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
		Fps:     int64(fps),
	})
	if err != nil {
		return nil, err
	}
	source.id = int(response.Id)
	source.start = time.Now()
	go source.sendFrameLoop()
	go source.monitorLoop()
	return source, nil
}

func (s *Source) monitorLoop() {
	timer := time.NewTicker(MONITOR_INTERVAL)
	for {
		fps := float64(s.sent) / time.Since(s.start).Seconds()
		logrus.Infof("Current frame: %d; Config FPS: %d fps; Actual FPS %.3f fps", s.currentIndex, s.currentFPS, fps)
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
				os.Exit(0)
			}
		}
	}
}

func (s *Source) sendFrame(current int) {
	file, err := os.Open(fmt.Sprintf("%s/%06d.jpg", s.datadir, current))
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
	s.sent++
}

func (s *Source) SetFramerate(ctx context.Context, request *ps.SetFramerateRequest) (*ps.SetFramerateResponse, error) {
	s.m.Lock()
	s.currentFPS = int(request.FrameRate)
	s.m.Unlock()
	return &ps.SetFramerateResponse{}, nil
}
