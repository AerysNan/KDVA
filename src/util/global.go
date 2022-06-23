package util

import (
	"errors"
	"time"
)

var (
	ErrDatasetNotFound = errors.New("dataset not found")
	ErrSourceNotFound  = errors.New("source ID not found")
	ErrEdgeNotFound    = errors.New("edge ID not found")
	ErrSourceExist     = errors.New("source ID already exist")
	ErrEdgeExist       = errors.New("edge ID already exist")
	ErrAddModel        = errors.New("failed to add model")
	ErrAddSource       = errors.New("failed to add source")
	ErrBuildConnection = errors.New("failed to establish connection")
	FrameDir           = "frames"
	ModelDir           = "models"
	ResultDir          = "results"
)

const (
	MONITOR_INTERVAL   = time.Millisecond * 1000
	CONNECTION_TIMEOUT = time.Second * 20
	QUEUE_TIMEOUT      = time.Millisecond * 100
)

type SimulationConfig struct {
	NEdges          int         `json:"n_edges"`
	NSourcesPerEdge int         `json:"n_sources_per_edge"`
	EdgeFPS         int         `json:"edge_fps"`
	UplinkFPS       int         `json:"uplink_fps"`
	RetrainWindow   int         `json:"retrain_window"`
	RetrainCfgs     map[int]int `json:"retrain_cfgs"`
	InferenceCfgs   map[int]int `json:"inference_cfgs"`
}

type DatasetInfo struct {
	Name      string `json:"name"`
	Size      int    `json:"size"`
	Framerate int    `json:"fps"`
}

type DatasetsInfo map[string]DatasetInfo

type SourceConfig struct {
	OriginalFramerate  int
	InferenceFramerate int
	UploadingFramerate int
	InfSamplePos       []int
	RetSamplePos       []int
}
