import argparse
import logging
import grpc
import json
import os

from distill import InferThread, DistillThread

import trainer_pb2
import trainer_pb2_grpc
import cloud_pb2_grpc

from mmdet.apis import init_detector
from concurrent import futures
from queue import Queue

DISTILL_INTERVAL = 1800
MAX_MESSAGE_LENGTH = 1 << 30


class Trainer(trainer_pb2_grpc.TrainerForCloudServicer):
    def __init__(self, model_name, gpu_name, client):
        models = json.load(open(f'{os.getcwd()}/data/model.json'))
        config_file = f"{os.getcwd()}/configs/{models[model_name]['config']}"
        checkpoint_file = f"{os.getcwd()}/checkpoints/{models[model_name]['checkpoint']}"
        model = init_detector(config_file, checkpoint_file, device=gpu_name)
        self.model = model
        self.client = client
        self.epoch_dict = {}
        self.queue_dict = {}

    def AddFrame(self, request, _):
        prefix = f'{request.edge}_{request.source}'
        if not prefix in self.epoch_dict:
            self.epoch_dict[prefix] = -1
            inference_queue = Queue(maxsize=10)
            inference_thread = InferThread(inference_queue, self.model)
            inference_thread.start()
            self.queue_dict[prefix] = inference_queue

        if request.index // DISTILL_INTERVAL > self.epoch_dict[prefix]:
            if self.epoch_dict[prefix] >= 0:
                print(
                    f'Prepare distillation {prefix} on epoch {self.epoch_dict[prefix]}')
                self.queue_dict[prefix].join()
                distill_thread = DistillThread(
                    self.client, prefix, self.epoch_dict[prefix])
                distill_thread.start()
            self.epoch_dict[prefix] += 1
            os.makedirs(
                f'dump/data/{prefix}/epoch_{self.epoch_dict[prefix]}', exist_ok=True)
            os.makedirs(
                f'dump/label/{prefix}/epoch_{self.epoch_dict[prefix]}', exist_ok=True)
        path = f'dump/data/{prefix}/epoch_{self.epoch_dict[prefix]}/{request.index:06d}.jpg'
        with open(path, 'wb') as f:
            f.write(request.content)
        self.queue_dict[prefix].put(path)
        return trainer_pb2.AddFrameResponse()

    def FetchModel(self, request, _):
        print(
            f'Fetch model request from edge {request.edge} source {request.source}')
        return trainer_pb2.FetchModelResponse(model=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--model', '-m', type=str, default='resnet101',
                        help='model use for inference')
    parser.add_argument('--port', '-p', type=str, default='8089',
                        help='model use for inference')
    parser.add_argument('--cloud', '-c', type=str, default='0.0.0.0:8088',
                        help='address of cloud server')
    parser.add_argument('--gpu', '-g', type=str, default='cuda:1',
                        help="name of GPU device to run inference")
    args = parser.parse_args()
    channel = grpc.insecure_channel(args.cloud,
                                    options=[
                                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                    ]
                                    )
    client = cloud_pb2_grpc.CloudForTrainerStub(channel)
    os.makedirs('dump/data', exist_ok=True)
    os.makedirs('dump/label', exist_ok=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trainer_pb2_grpc.add_TrainerForCloudServicer_to_server(
        Trainer(args.model, args.gpu, client), server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Training server started at {listen_address}')
    server.start()
    server.wait_for_termination()
