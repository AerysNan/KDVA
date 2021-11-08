PROTOS = source edge cloud worker
BINS = source edge cloud

gen:
	for proto in $(PROTOS) ; do \
		protoc --go_out=proto/$$proto --go-grpc_out=proto/$$proto proto/$$proto/$$proto.proto ; \
		python3 -m grpc_tools.protoc -Iproto/$$proto --python_out=proto/$$proto --grpc_python_out=proto/$$proto proto/$$proto/$$proto.proto ; \
	done

build:
	for bin in $(BINS) ; do \
		go build -o bin/$$bin cmd/$$bin/main.go ; \
	done