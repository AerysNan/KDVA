import grpc
import logging
import argparse
import concurrent.futures as futures

import trainer_pb2_grpc
import teacher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--teacher-checkpoint', '-tk', type=str, help='teacher model checkpoint', required=True)
    parser.add_argument('--student-checkpoint', '-sk', type=str, help='student model checkpoint', required=True)
    parser.add_argument('--teacher-config', '-tc', type=str, help='teacher model configuration', required=True)
    parser.add_argument('--student-config', '-sc', type=str, help='student model configuration', required=True)
    parser.add_argument('--emulation-config', '-e', type=str, help='emulation configuration', default=None)
    parser.add_argument('--device', '-d', type=str, help='device', default='cuda:0')
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
