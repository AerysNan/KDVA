import grpc
import student
import logging
import argparse
import worker_pb2_grpc
from concurrent import futures

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--model', '-m', type=str, default='ssd',
                        help='model use for inference')
    parser.add_argument('--port', '-p', type=str, default='8086',
                        help='model use for inference')
    parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                        help="name of GPU device to run inference")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_pb2_grpc.add_WorkerForEdgeServicer_to_server(
        student.Student(args.model, args.gpu), server)
    listen_address = f'0.0.0.0:{args.port}'
    server.add_insecure_port(listen_address)
    logging.info(f'Inference worker started at {listen_address}')
    server.start()
    server.wait_for_termination()
