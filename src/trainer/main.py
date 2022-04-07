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
    parser.add_argument('--teacher-gpu', '-tg', type=str, default='cuda:1', help='teacher model GPU device')
    parser.add_argument('--student-gpu', '-sg', type=str, default='cuda:0', help='student_model GPU device')
    parser.add_argument('--emulated', '-e', action='store_true', help='run trainer in emulated mode')
    parser.add_argument('--port', '-p', type=str, default='8089', help='listening port')
    parser.add_argument('--cloud', '-c', type=str, default='0.0.0.0:8088', help='address of cloud server')
    args = parser.parse_args()
    channel = grpc.insecure_channel(args.cloud, options=[
        ('grpc.max_send_message_length', 1 << 30),
        ('grpc.max_receive_message_length', 1 << 30),
    ])
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    teacher = teacher.Teacher(**args.__dict__)
    trainer_pb2_grpc.add_TrainerForCloudServicer_to_server(teacher, server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Training server started at {listen_address}')
    server.start()
    server.wait_for_termination()
