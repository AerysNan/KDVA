GO_PROTOS = edge cloud worker trainer
PY_PROTOS = worker trainer cloud
BINS = source edge cloud
PYS = worker trainer
gen:
	for proto in $(GO_PROTOS) ; do \
		protoc --go_out=proto/$$proto --go-grpc_out=proto/$$proto proto/$$proto/$$proto.proto ; \
	done
	for proto in $(PY_PROTOS) ; do \
		python3 -m grpc_tools.protoc -I proto/$$proto --python_out=$$proto --grpc_python_out=$$proto proto/$$proto/$$proto.proto ; \
	done
build:
	for bin in $(BINS) ; do \
		CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o cmd/$$bin/$$bin cmd/$$bin/main.go ; \
	done
docker:
	for bin in $(BINS) ; do \
		docker build -t aerysnan/kdva_$$bin:latest cmd/$$bin ; \
	done
	for py in $(PYS) ; do \
		docker build -t aerysnan/kdva_$$py:latest $$py ; \
	done
push:
	for d in $(BINS) $(PYS) ; do \
		docker push aerysnan/kdva_$$d:latest ; \
	done