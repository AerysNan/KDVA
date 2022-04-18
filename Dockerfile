ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget zip tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch${PYTORCH}/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install Golang
WORKDIR /root
RUN wget https://go.dev/dl/go1.18.1.linux-amd64.tar.gz
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protoc-3.20.0-linux-x86_64.zip
RUN tar xvf go1.18.1.linux-amd64.tar.gz
RUN unzip protoc-3.20.0-linux-x86_64.zip -d protoc
RUN rm go1.18.1.linux-amd64.tar.gz protoc-3.20.0-linux-x86_64.zip
ENV PATH="/root/go/bin:/root/protoc/bin:${PATH}"
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2


# Install project
RUN git clone https://github.com/AerysNan/KDVA.git /root/KDVA
WORKDIR /root/KDVA/src
RUN go mod tidy
RUN pip install --no-cache-dir grpcio-tools grpcio matplotlib
RUN make gen
RUN make build