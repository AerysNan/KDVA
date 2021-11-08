package main

import (
	"fmt"
	"net"
	"vakd/edge"
	pe "vakd/proto/edge"
	pw "vakd/proto/worker"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	port      = kingpin.Flag("port", "listen port of edge server").Short('p').Default("8084").String()
	inference = kingpin.Flag("worker", "address of inference worker").Short('i').Default("0.0.0.0:8086").String()
	debug     = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	inferenceConnection, err := grpc.Dial(*inference, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Connect to infernce server %s failed", *inference)
	}
	defer inferenceConnection.Close()
	inferenceClient := pw.NewWorkerForEdgeClient(inferenceConnection)
	edgeServer, err := edge.NewEdge(inferenceClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Create edge server failed")
	}
	listenAddress := fmt.Sprintf("0.0.0.0:%s", *port)
	listen, err := net.Listen("tcp", listenAddress)
	if err != nil {
		logrus.WithError(err).Fatalf("Listen to port %s failed", *port)
	}
	server := grpc.NewServer()
	pe.RegisterEdgeForSourceServer(server, edgeServer)
	logrus.Infof("Edge server started on address %s", listenAddress)
	if err = server.Serve(listen); err != nil {
		logrus.WithError(err).Fatal("Edge server failed")
	}
}
