package main

import (
	"context"
	pe "vakd/proto/edge"
	"vakd/util"

	"vakd/source"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	id      = kingpin.Flag("id", "source ID").Short('i').Required().Int()
	config  = kingpin.Flag("config", "dataset configuration file").Short('c').Default("cfg_data.json").String()
	dataset = kingpin.Flag("dataset", "dataset name of source").Short('d').Required().String()
	datadir = kingpin.Flag("dir", "dataset directory").Short('p').Default("data").String()
	edge    = kingpin.Flag("edge", "address of edge").Short('e').Default("0.0.0.0:8084").String()
	debug   = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	ctx, cancel := context.WithTimeout(context.Background(), util.CONNECTION_TIMEOUT)
	defer cancel()
	edgeConnection, err := grpc.DialContext(ctx, *edge, grpc.WithBlock(), grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Connect to edge server %s failed", *edge)
	}
	defer edgeConnection.Close()
	edgeClient := pe.NewEdgeForSourceClient(edgeConnection)
	s, err := source.NewSource(*id, *dataset, *datadir, *config, edgeClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Create source server failed")
	}
	logrus.Infof("Source started")
	s.Start()
	logrus.Infof("Source exited")
}
