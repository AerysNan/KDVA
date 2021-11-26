import argparse
import logging
import grpc
import json
import ast
import os

from distill import FakeDistillThread, DistillThread
from infer import InferThread

import trainer_pb2
import trainer_pb2_grpc
import cloud_pb2_grpc

from mmdet.apis import init_detector
from concurrent import futures
from queue import Queue

MAX_MESSAGE_LENGTH = 1 << 30


class Trainer(trainer_pb2_grpc.TrainerForCloudServicer):
    def __init__(self, model_name, gpu_name, client, train, distill_interval, monitor_interval, config):
        models = json.load(open(f'{os.getcwd()}/data/model.json'))
        config_file = f"{os.getcwd()}/configs/{models[model_name]['config']}"
        checkpoint_file = f"{os.getcwd()}/checkpoints/{models[model_name]['checkpoint']}"
        model = init_detector(config_file, checkpoint_file, device=gpu_name)
        self.model = model
        self.client = client
        self.train = train
        self.distill_interval = distill_interval
        self.monitor_interval = monitor_interval
        self.config = config
        self.epoch_dict = {}
        self.queue_dict = {}

    def SendFrame(self, request, _):
        prefix = f'{request.edge}_{request.source}'
        if not prefix in self.epoch_dict:
            self.epoch_dict[prefix] = -1
            inference_queue = Queue(maxsize=10)
            inference_thread = InferThread(inference_queue, self.model, self.monitor_interval, self.config, self.client)
            inference_thread.start()
            self.queue_dict[prefix] = inference_queue
        if request.index // self.distill_interval > self.epoch_dict[prefix]:
            if self.epoch_dict[prefix] >= 0:
                print(f'Prepare distillation {prefix} on epoch {self.epoch_dict[prefix]}')
                self.queue_dict[prefix].join()
                distill_thread = None
                if self.train:
                    distill_thread = DistillThread(self.client, request.edge, request.source, self.epoch_dict[prefix])
                else:
                    distill_thread = FakeDistillThread(self.client, request.edge, request.source, self.epoch_dict[prefix], request.dataset)
                distill_thread.start()
            self.epoch_dict[prefix] += 1
            os.makedirs(f'dump/data/{prefix}/epoch_{self.epoch_dict[prefix]}', exist_ok=True)
            os.makedirs(f'dump/label/{prefix}/epoch_{self.epoch_dict[prefix]}', exist_ok=True)
            os.makedirs(f'dump/fake/{prefix}/epoch_{self.epoch_dict[prefix]}', exist_ok=True)
        with open(f'dump/data/{prefix}/epoch_{self.epoch_dict[prefix]}/{request.index:06d}.jpg', 'wb') as f:
            f.write(request.content)
        with open(f'dump/fake/{prefix}/epoch_{self.epoch_dict[prefix]}/{request.index:06d}.pkl', 'wb') as f:
            f.write(request.annotation)
        self.queue_dict[prefix].put({
            "edge": request.edge,
            "source": request.source,
            "epoch": self.epoch_dict[prefix],
            "index": request.index
        })
        return trainer_pb2.CloudSendFrameResponse()


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
                        help='name of GPU device to run inference')
    parser.add_argument('--train', '-t', type=ast.literal_eval, default='False',
                        help='whether to use online trained models')
    parser.add_argument('--distill-interval', type=int, default=500,
                        help='length of the distillation interval')
    parser.add_argument('--monitor-interval', type=int, default=1000,
                        help='length of the distillation interval')
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')

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
    os.makedirs('dump/fake', exist_ok=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trainer = Trainer(
        model_name=args.model,
        gpu_name=args.gpu,
        client=client,
        train=args.train,
        distill_interval=args.distill_interval,
        monitor_interval=args.monitor_interval,
        config=args.config
    )
    trainer_pb2_grpc.add_TrainerForCloudServicer_to_server(trainer, server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Training server started at {listen_address}')
    server.start()
    server.wait_for_termination()
