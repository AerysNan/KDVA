import grpc
import logging
import argparse
import concurrent.futures as futures

import trainer_pb2_grpc
import teacher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--teacher-model', '-tm', type=str, default='resnet101', help='teacher model name')
    parser.add_argument('--student-model', '-sm', type=str, default='ssd', help='student model name')
    parser.add_argument('--emulated', '-e', action='store_true', help='run trainer in emulated mode')
    parser.add_argument('--port', '-p', type=str, default='8089', help='listening port')
    args = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    teacher = teacher.Trainer(**args.__dict__)
    trainer_pb2_grpc.add_TrainerForCloudServicer_to_server(teacher, server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Training server started at {listen_address}')
    server.start()
    server.wait_for_termination()
