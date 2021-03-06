import grpc
import student
import logging
import argparse
import worker_pb2_grpc
from concurrent import futures

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--config', '-c', type=str, help='model config used for inference', required=True)
    parser.add_argument('--checkpoint', '-k', type=str, help='model checkpoint used for inference', required=True)
    parser.add_argument('--port', '-p', type=str, default='8086', help='listening port for worker')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help="name of GPU device to run inference")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_pb2_grpc.add_WorkerForEdgeServicer_to_server(student.Student(**args.__dict__), server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Inference worker started at {listen_address}')
    server.start()
    server.wait_for_termination()
