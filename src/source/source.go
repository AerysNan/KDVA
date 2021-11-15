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
	ORINGAL_FRAMERATE = 30
	MONITOR_INTERVAL  = time.Millisecond * 1000
)

type Source struct {
	ps.SourceForEdgeServer
	id      int
	current int
	fps     int
	count   int
	sent    int
	start   time.Time
	datadir string
	client  pe.EdgeForSourceClient
	m       sync.RWMutex
}

func NewSource(path string, client pe.EdgeForSourceClient, fps int) (*Source, error) {
	source := &Source{
		m:       sync.RWMutex{},
		current: 0,
		sent:    0,
		start:   time.Now(),
		fps:     fps,
		datadir: path,
		client:  client,
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
	response, err := client.Register(context.Background(), &pe.RegisterRequest{})
	if err != nil {
		return nil, err
	}
	source.id = int(response.Id)
	go source.sendFrameLoop()
	go source.monitorLoop()
	return source, nil
}

func (s *Source) monitorLoop() {
	timer := time.NewTicker(MONITOR_INTERVAL)
	for {
		fps := float64(s.sent) / time.Since(s.start).Seconds()
		logrus.Infof("Current frame: %d; Config FPS: %d fps; Actual FPS %.3f fps", s.current, s.fps, fps)
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
		s.m.RLock()
		duration = time.Second / time.Duration(s.fps)
		stride := ORINGAL_FRAMERATE / s.fps
		s.m.RUnlock()
		timer.Reset(duration)
		s.sendFrame(s.current)
		s.current += stride
		if s.current >= s.count {
			logrus.Warnf("Stream %d exited since all frames are sent to edge", s.id)
			s.client.Terminate(context.Background(), &pe.TerminateRequest{
				Id: int64(s.id),
			})
			os.Exit(0)
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
	_, err = s.client.SendFrame(context.Background(), &pe.SendFrameRequest{
		Id:      int64(s.id),
		Current: int64(current),
		Content: content,
	})
	if err != nil {
		logrus.WithError(err).Errorf("Send frame %d failed", current)
	}
	s.sent++
}

func (s *Source) SetFramerate(ctx context.Context, request *ps.SetFramerateRequest) (*ps.SetFramerateResponse, error) {
	s.m.Lock()
	s.fps = int(request.FrameRate)
	s.m.Unlock()
	return &ps.SetFramerateResponse{}, nil
}
