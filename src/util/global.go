package util

import (
	"errors"
	"time"
)

var (
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
	MONITOR_INTERVAL            = time.Millisecond * 1000
	CLOSE_INTERVAL              = time.Millisecond * 1000
	TIMEOUT_DURATION            = time.Millisecond * 100
	INITIAL_RETRAIN_WINDOW      = time.Second * 60
	COMPUTATION_BOTTLENECK      = 100
	UPLINK_NETWORK_BOTTLENECK   = 100
	DOWNLINK_NETWORK_BOTTLENECK = 50
)

const (
	SOURCE_STATUS_CONNECTED = iota
	SOURCE_STATUS_DISCONNECTED
	SOURCE_STATUS_CLOSED
)

type SimulationConfig struct {
	NEdges                         int `json:"n_edges"`
	NSourcesPerEdge                int `json:"n_sources_per_edge"`
	EdgeResourceBottleneckFPS      int `json:"edge_resource_bottleneck_fps"`
	UplinkResourceBottleneckFPS    int `json:"uplink_resource_bottleneck_fps"`
	DownlinkResourceBottleneckMbPS int `json:"downlink_resource_bottleneck_mbps"`
	ModelSizeMb                    int `json:"model_size_mb"`
}
