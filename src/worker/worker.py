import argparse
import logging
import numpy as np
import grpc
import json
import cv2
import os

import worker_pb2
import worker_pb2_grpc

from mmdet.apis import init_detector, inference_detector
from concurrent import futures


class Worker(worker_pb2_grpc.WorkerForEdgeServicer):
    def __init__(self, model_name, gpu_name):
        models = json.load(open(f'{os.getcwd()}/data/model.json'))
        config_file = f"{os.getcwd()}/configs/{models[model_name]['config']}"
        checkpoint_file = f"{os.getcwd()}/checkpoints/{models[model_name]['checkpoint']}"
        model = init_detector(config_file, checkpoint_file, device=gpu_name)
        self.model = model

    def Infer(self, request, _):
        decoded = cv2.imdecode(np.frombuffer(request.content, np.uint8), -1)
        class_results = inference_detector(self.model, decoded)
        result = []
        for class_result in class_results:
            boxes = []
            for box in class_result:
                boxes.append(worker_pb2.BoundingBox(params=box))
            result.append(worker_pb2.ClassResult(boxes=boxes))
        return worker_pb2.InferResponse(result=result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--model', '-m', type=str, required=True,
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
