package main

import (
	"fmt"
	"net"
	"vakd/cloud"
	pc "vakd/proto/cloud"
	pt "vakd/proto/trainer"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	port    = kingpin.Flag("port", "listen port for cloud").Short('p').Default("8088").String()
	workDir = kingpin.Flag("work-dir", "work directory of cloud").Short('d').Required().String()
	config  = kingpin.Flag("config", "simulation configuration file").Short('c').Default("configs.json").String()
	trainer = kingpin.Flag("trainer", "address of trainer").Short('t').Default("0.0.0.0:8089").String()
	debug   = kingpin.Flag("debug", "use debug level of logging").Default("false").Bool()
)

func main() {
	kingpin.Parse()
	if *debug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("Set log level to debug")
	}
	trainerConnection, err := grpc.Dial(*trainer, grpc.WithInsecure())
	if err != nil {
		logrus.WithError(err).Fatalf("Connect to training server %s failed", *trainer)
	}
	defer trainerConnection.Close()
	trainerClient := pt.NewTrainerForCloudClient(trainerConnection)
	listenAddress := fmt.Sprintf("0.0.0.0:%s", *port)
	cloudServer, err := cloud.NewCloud(listenAddress, *workDir, *config, trainerClient)
	if err != nil {
		logrus.WithError(err).Fatalf("Create cloud server failed")
	}
	listen, err := net.Listen("tcp", listenAddress)
	if err != nil {
		logrus.WithError(err).Fatalf("Listen to port %s failed", *port)
	}
	server := grpc.NewServer(grpc.MaxRecvMsgSize(1 << 30))
	pc.RegisterCloudForEdgeServer(server, cloudServer)
	logrus.Infof("Cloud server started on address %s", listenAddress)
	if err = server.Serve(listen); err != nil {
		logrus.WithError(err).Fatal("Cloud server failed")
	}
}
