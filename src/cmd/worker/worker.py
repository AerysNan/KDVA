import argparse
import logging
import pickle
import numpy as np
import grpc
import json
import mmcv
import cv2
import os

import rwlock
import worker_pb2
import worker_pb2_grpc

from mmdet.apis import init_detector, inference_detector
from concurrent import futures

lock = rwlock.RWLock()


class Worker(worker_pb2_grpc.WorkerForEdgeServicer):
    def __init__(self, model_name, gpu_name):
        models = json.load(open(f'{os.getcwd()}/data/model.json'))
        self.config_file = mmcv.Config.fromfile(f"{os.getcwd()}/configs/{models[model_name]['config']}")
        self.checkpoint_file = f"{os.getcwd()}/checkpoints/{models[model_name]['checkpoint']}"
        self.gpu_name = gpu_name
        self.model_dict = {}

    def AddModel(self, request, _):
        logging.info(f'Add new model for S{request.source}')
        self.model_dict[request.source] = init_detector(
            self.config_file, self.checkpoint_file, self.gpu_name)
        return worker_pb2.AddModelResponse()

    def RemoveModel(self, request, _):
        logging.info(f'Remove model for S{request.source}')
        del self.model_dict[request.source]
        return worker_pb2.RemoveModelResponse()

    def UpdateModel(self, request, _):
        global lock
        logging.info(f'Update model S{request.source} at path {request.path}')
        model = init_detector(self.config_file, request.path, self.gpu_name)
        lock.w_acquire()
        self.model_dict[request.source] = model
        lock.w_release()
        return worker_pb2.EdgeUpdateModelResponse()

    def InferFrame(self, request, _):
        global lock
        decoded = cv2.imdecode(np.frombuffer(request.content, np.uint8), -1)
        lock.r_acquire()
        class_results = inference_detector(
            self.model_dict[request.source], decoded)
        lock.r_release()
        result = pickle.dumps(class_results)
        return worker_pb2.InferFrameResponse(result=result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--model', '-m', type=str, default='mobilenet',
                        help='model use for inference')
    parser.add_argument('--port', '-p', type=str, default='8086',
                        help='model use for inference')
    parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                        help="name of GPU device to run inference")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_pb2_grpc.add_WorkerForEdgeServicer_to_server(
        Worker(args.model, args.gpu), server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Inference worker started at {listen_address}')
    server.start()
    server.wait_for_termination()
