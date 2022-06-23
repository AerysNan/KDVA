package main

import (
	"context"
	"fmt"
	"net"
	"vakd/edge"
	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pw "vakd/proto/worker"
	"vakd/util"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	id     = kingpin.Flag("id", "edge ID").Short('i').Required().Int()
	port   = kingpin.Flag("port", "listen port for edge").Short('p').Default("8084").String()
	dir    = kingpin.Flag("dir", "work directory of edge").Short('d').Required().String()
	config = kingpin.Flag("config", "simulation configuration file").Short('c').Default("configs.json").String()
	worker = kingpin.Flag("worker", "address of worker").Short('w').Default("0.0.0.0:8086").String()
	cloud  = kingpin.Flag("cloud", "address of cloud").Short('a').Default("0.0.0.0:8088").String()
	debug  = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	ctx, cancel := context.WithTimeout(context.Background(), util.CONNECTION_TIMEOUT)
	defer cancel()
	inferenceConnection, err := grpc.DialContext(ctx, *worker, grpc.WithBlock(), grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to connect to infernce server %s", *worker)
	}
	defer inferenceConnection.Close()
	workerClient := pw.NewWorkerForEdgeClient(inferenceConnection)
	ctx, cancel = context.WithTimeout(context.Background(), util.CONNECTION_TIMEOUT)
	defer cancel()
	cloudConnection, err := grpc.DialContext(ctx, *cloud, grpc.WithBlock(), grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to connect to cloud server %s 5 times", *cloud)
	}
	defer cloudConnection.Close()
	cloudClient := pc.NewCloudForEdgeClient(cloudConnection)
	listenAddress := fmt.Sprintf("0.0.0.0:%s", *port)
	edgeServer, err := edge.NewEdge(*id, listenAddress, *dir, *config, workerClient, cloudClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to create edge server")
	}
	listen, err := net.Listen("tcp", listenAddress)
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to listen at port %s", *port)
	}
	server := grpc.NewServer(grpc.MaxRecvMsgSize(1<<30), grpc.MaxSendMsgSize(1<<30))
	pe.RegisterEdgeForSourceServer(server, edgeServer)
	pe.RegisterEdgeForCloudServer(server, edgeServer)
	logrus.Infof("Edge server started on address %s", listenAddress)
	if err = server.Serve(listen); err != nil {
		logrus.WithError(err).Fatal("Edge server failed")
	}
}
