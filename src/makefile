GO_PROTOS = source edge cloud worker trainer
PY_PROTOS = worker trainer cloud
BINS = source edge cloud

gen:
	for proto in $(GO_PROTOS) ; do \
		protoc --go_out=proto/$$proto --go-grpc_out=proto/$$proto proto/$$proto/$$proto.proto ; \
	done
	for proto in $(PY_PROTOS) ; do \
		python3 -m grpc_tools.protoc -Iproto/$$proto --python_out=cmd/$$proto --grpc_python_out=cmd/$$proto proto/$$proto/$$proto.proto ; \
	done
	mv cmd/cloud/*.py cmd/trainer
build:
	for bin in $(BINS) ; do \
		go build -o ../bin/$$bin cmd/$$bin/main.go ; \
	done
clean:
	find . -name '*.pb.go' -type f -delete
	find . -name '*_pb2*.py' -type f -delete