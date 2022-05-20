from collections import defaultdict
import logging
import shutil
import pickle
import torch
import json
import copy
import mmcv
import cv2
import os

import trainer_pb2
import trainer_pb2_grpc
from rwlock import RWLock

from mmdet.apis import init_detector, inference_detector, train_detector, set_random_seed
from mmdet.datasets.builder import build_dataset
from mmdet.models import build_detector


LOCK = RWLock()
MODEL_FILE = 'models.json'
TEMPLATE_FILE = 'template.json'
ANNOTATION_DIR = 'annotations'
LABEL_DIR = 'labels'
FRAME_DIR = 'frames'
MODEL_DIR = 'models'
BACKUP_DIR = 'backups'
RETRAIN_DIR = 'retrain'
RETRAIN_THRESHOLD = 10
IMG_HEIGHT = 540
IMG_WIDTH = 960


class Teacher(trainer_pb2_grpc.TrainerForCloudServicer):
    def __init__(self, student_model, teacher_model, student_gpu, teacher_gpu, emulated, **_):
        models = json.load(open(MODEL_FILE))
        self.emulated = emulated
        self.teacher_config = mmcv.Config.fromfile(models[teacher_model]['config'])
        self.student_config = mmcv.Config.fromfile(models[student_model]['config'])
        self.teacher_checkpoint = models[teacher_model]['checkpoint']
        self.student_checkpoint = models[student_model]['checkpoint']
        self.teacher_model = init_detector(self.teacher_config, self.teacher_checkpoint, device=teacher_gpu)
        self.teacher_gpu = teacher_gpu
        self.student_gpu = student_gpu
        self.student_dict = {}
        self.frame_dict = defaultdict(lambda: [])

    def InitTrainer(self, request, _):
        self.work_dir = request.work_dir
        logging.info(f'Work directory set to {self.work_dir}')
        os.makedirs(f'{self.work_dir}/{ANNOTATION_DIR}', exist_ok=True)
        os.makedirs(f'{self.work_dir}/{BACKUP_DIR}', exist_ok=True)
        os.makedirs(f'{self.work_dir}/{LABEL_DIR}', exist_ok=True)
        os.makedirs(f'{self.work_dir}/{FRAME_DIR}', exist_ok=True)
        os.makedirs(f'{self.work_dir}/{MODEL_DIR}', exist_ok=True)
        return trainer_pb2.InitTrainerResponse()

    def SendFrame(self, request, _):
        model_key = (request.edge, request.source)
        if model_key not in self.student_dict:
            logging.info(f'Add new stream of source {request.source} edge {request.edge}')
            self.student_dict[model_key] = self.init_model()
        logging.debug(f'Receive frame {request.index} of source {request.source} edge {request.edge}')
        frame = cv2.imread(self.get_frame_dir(request.source, request.index))
        LOCK.r_acquire()
        results = inference_detector(self.student_dict[model_key], frame)
        LOCK.r_release()
        with open(self.get_result_dir(request.edge, request.source, request.index), 'wb') as f:
            pickle.dump(results, f)
        labels = inference_detector(self.teacher_model, frame)
        with open(self.get_label_dir(request.edge, request.source, request.index), 'wb') as f:
            pickle.dump(labels, f)
        LOCK.w_acquire()
        self.frame_dict[(request.edge, request.source)].append(request.index)
        LOCK.w_release()
        return trainer_pb2.CloudSendFrameResponse()

    def TriggerRetrain(self, request, _):
        LOCK.w_acquire()
        indices = copy.deepcopy(self.frame_dict[(request.edge, request.source)])
        if len(indices) < RETRAIN_THRESHOLD:
            LOCK.w_release()
            return trainer_pb2.TriggerRetrainResponse(updated=False)
        self.frame_dict[(request.edge, request.source)] = []
        LOCK.w_release()
        template = self.generate_annotation(request.edge, request.source, indices)
        retrain_dir = self.get_retrain_dir(request.edge, request.source, request.version)
        os.makedirs(retrain_dir, exist_ok=True)
        with open(f'{retrain_dir}/{TEMPLATE_FILE}', 'w') as f:
            json.dump(template, f)
        cfg = copy.deepcopy(self.student_config)
        cfg.data.train.ann_file = f'{retrain_dir}/{TEMPLATE_FILE}'
        cfg.data.train.img_prefix = ''
        cfg.log_level = 'WARN'
        cfg.work_dir = retrain_dir
        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.gpu_ids = [i for i in range(torch.cuda.device_count())]
        cfg.seed = 0
        set_random_seed(cfg.seed, True)
        dataset = build_dataset(cfg.data.train)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        train_detector(model, dataset, cfg)
        shutil.move(f'{retrain_dir}/latest.pth', self.get_model_dir(request.edge, request.source, request.version))
        # TODO: fill in retrain profile
        return trainer_pb2.TriggerRetrainResponse(ret_profile=None, updated=True)

    def init_model(self):
        return init_detector(self.student_config, self.student_checkpoint, device=self.get_available_gpu())

    def get_result_dir(self, edge, source, index):
        return f'{self.work_dir}/{ANNOTATION_DIR}/{edge}-{source}-{index}.pkl'

    def get_frame_dir(self, edge, source, index):
        return f'{self.work_dir}/{FRAME_DIR}/{edge}-{source}-{index:06d}.jpg'

    def get_label_dir(self, edge, source, index):
        return f'{self.work_dir}/{LABEL_DIR}/{edge}-{source}-{index}.pkl'

    def get_model_dir(self, edge, source, version):
        return f'{self.work_dir}/{MODEL_DIR}/{edge}-{source}-{version}.pth'

    def get_retrain_dir(self, edge, source, version):
        return f'{self.work_dir}/{RETRAIN_DIR}/{edge}-{source}-{version}'

    def get_available_gpu():
        return 'cuda:0'

    def generate_annotation(self, edge, source, indices, threshold=0.5):
        indices.sort()
        with open(TEMPLATE_FILE) as f:
            template = json.load(f)
        for i, index in enumerate(indices):
            template['images'].append({
                'id': i,
                'file_name': self.get_frame_dir(edge, source, index),
                'height': IMG_HEIGHT,
                'width': IMG_WIDTH,
            })
            with open(self.get_label_dir(edge, source, index), 'rb') as f:
                labels = pickle.load(f)
            uid = 0
            for j, label in enumerate(labels):
                for bbox in label:
                    if bbox[4] < threshold:
                        continue
                    converted = self.xyxy2xywh(bbox.tolist())
                    template['annotations'].append({
                        'id': uid,
                        'image_id': i,
                        'category_id': j,
                        'iscrowd': 0,
                        'bbox': converted,
                        'area': converted[2] * converted[3]
                    })
                    uid += 1
        return template

    def xyxy2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
