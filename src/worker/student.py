import logging
import pickle
import json
import mmcv
import cv2

import rwlock
import worker_pb2
import worker_pb2_grpc

from mmdet.apis import init_detector, inference_detector

LOCK = rwlock.RWLock()
MODEL_FILE = 'models.json'
RESULT_DIR = 'results'
FRAME_DIR = 'frames'
MODEL_DIR = 'models'


class Student(worker_pb2_grpc.WorkerForEdgeServicer):
    def __init__(self, model_name, gpu_name):
        with open(MODEL_FILE) as f:
            models = json.load(f)
        self.config = mmcv.Config.fromfile(models[model_name]['config'])
        self.base_model = models[model_name]['checkpoint']
        self.gpu_name = gpu_name
        self.model_dict = {}

    def InitWorker(self, request, _):
        self.work_dir = request.work_dir
        logging.info(f'Set work directory to {request.work_dir}')
        return worker_pb2.InitWorkerResponse()

    def RemoveModel(self, request, _):
        logging.info(f'Remove model for source {request.source}')
        del self.model_dict[request.source]
        return worker_pb2.RemoveModelResponse()

    def UpdateModel(self, request, _):
        global LOCK
        logging.info(f'Update model of source {request.source} to version {request.version}')
        path = self.get_model_dir(request.source, request.version) if request.version > 0 else self.base_model
        model = init_detector(self.config, path, self.gpu_name)
        LOCK.w_acquire()
        self.model_dict[request.source] = model
        LOCK.w_release()
        return worker_pb2.EdgeUpdateModelResponse()

    def InferFrame(self, request, _):
        global LOCK
        decoded = cv2.imread(self.get_frame_dir(request.source, request.index))
        LOCK.r_acquire()
        results = inference_detector(
            self.model_dict[request.source], decoded)
        LOCK.r_release()
        with open(self.get_result_dir(request.source, request.index), 'w') as f:
            pickle.dump(results)
        return worker_pb2.InferFrameResponse()

    def get_model_dir(self, source, version):
        return f'{self.work_dir}/{MODEL_DIR}/{source}-{version}.pth'

    def get_frame_dir(self, source, index):
        return f'{self.work_dir}/{FRAME_DIR}/{source}-{index:06d}.jpg'

    def get_result_dir(self, source, index):
        return f'{self.work_dir}/{RESULT_DIR}/{source}-{index:06d}.jpg'
