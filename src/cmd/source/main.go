package main

import (
	"fmt"
	"net"
	pe "vakd/proto/edge"
	ps "vakd/proto/source"

	"vakd/source"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	port  = kingpin.Flag("port", "listen port of source server").Short('p').Default("8080").String()
	dir   = kingpin.Flag("dir", "data directory of source server").Short('d').Required().String()
	edge  = kingpin.Flag("edge", "address of edge server").Short('e').Default("0.0.0.0:8084").String()
	debug = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	edgeConnection, err := grpc.Dial(*edge, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Connect to edge server %s failed", *edge)
	}
	defer edgeConnection.Close()
	edgeClient := pe.NewEdgeForSourceClient(edgeConnection)
	sourceServer, err := source.NewSource(*dir, edgeClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Create source server failed")
	}
	listenAddress := fmt.Sprintf("0.0.0.0:%s", *port)
	listen, err := net.Listen("tcp", listenAddress)
	if err != nil {
		logrus.WithError(err).Fatalf("Listen to port %s failed", *port)
	}
	server := grpc.NewServer()
	ps.RegisterSourceForEdgeServer(server, sourceServer)
	logrus.Infof("Source server started on address %s", listenAddress)
	if err = server.Serve(listen); err != nil {
		logrus.WithError(err).Fatal("Source server failed")
	}
}
