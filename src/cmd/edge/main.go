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
	port   = kingpin.Flag("port", "listen port of edge server").Short('p').Default("8084").String()
	worker = kingpin.Flag("worker", "address of inference worker").Short('w').Default("0.0.0.0:8086").String()
	cloud  = kingpin.Flag("cloud", "address of training cloud").Short('c').Default("0.0.0.0:8088").String()
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
		logrus.WithError(err).Fatalf("Connect to infernce server %s failed", *worker)
	}
	defer inferenceConnection.Close()
	workerClient := pw.NewWorkerForEdgeClient(inferenceConnection)
	cloudConnection, err := grpc.Dial(*cloud, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Connect to cloud server %s failed", *cloud)
	}
	defer cloudConnection.Close()
	cloudClient := pc.NewCloudForEdgeClient(cloudConnection)
	listenAddress := fmt.Sprintf("0.0.0.0:%s", *port)
	edgeServer, err := edge.NewEdge(listenAddress, workerClient, cloudClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Create edge server failed")
	}
	listen, err := net.Listen("tcp", listenAddress)
	if err != nil {
		logrus.WithError(err).Fatalf("Listen to port %s failed", *port)
	}
	server := grpc.NewServer(grpc.MaxRecvMsgSize(1<<30), grpc.MaxSendMsgSize(1<<30))
	pe.RegisterEdgeForSourceServer(server, edgeServer)
	pe.RegisterEdgeForCloudServer(server, edgeServer)
	logrus.Infof("Edge server started on address %s", listenAddress)
	if err = server.Serve(listen); err != nil {
		logrus.WithError(err).Fatal("Edge server failed")
	}
}
