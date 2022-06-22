package main

import (
	"fmt"
	"net"
	"vakd/edge"
	pc "vakd/proto/cloud"
	pe "vakd/proto/edge"
	pw "vakd/proto/worker"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	id     = kingpin.Flag("id", "edge ID").Short('i').Required().Int()
	port   = kingpin.Flag("port", "listen port of edge server").Short('p').Default("8084").String()
	dir    = kingpin.Flag("dir", "work directory").Short('d').Required().String()
	config = kingpin.Flag("config", "configuration file of cloud server").Short('c').Default("configs.json").String()
	worker = kingpin.Flag("worker", "address of inference worker").Short('w').Default("0.0.0.0:8086").String()
	cloud  = kingpin.Flag("cloud", "address of training cloud").Short('a').Default("0.0.0.0:8088").String()
	debug  = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	inferenceConnection, err := grpc.Dial(*worker, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to connect to infernce server %s", *worker)
	}
	defer inferenceConnection.Close()
	workerClient := pw.NewWorkerForEdgeClient(inferenceConnection)
	cloudConnection, err := grpc.Dial(*cloud, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Failed to connect to cloud server %s", *cloud)
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
