from mmdet.apis import init_detector, inference_detector, train_detector, set_random_seed
from mmdet.datasets.builder import build_dataset
from mmdet.models import build_detector
from collections import defaultdict

import logging
import shutil
import pickle
import time
import json
import copy
import mmcv
import cv2
import os

import trainer_pb2
import trainer_pb2_grpc
from rwlock import RWLock
from template import TEMPLATE

LOCK = RWLock()

ANNO_DIR = 'annos'
LABEL_DIR = 'labels'
FRAME_DIR = 'frames'
MODEL_DIR = 'models'
RETRAIN_DIR = 'retrains'

IMG_HEIGHT = 720
IMG_WIDTH = 1280


class Trainer(trainer_pb2_grpc.TrainerForCloudServicer):
    def __init__(self, teacher_checkpoint, teacher_config, student_checkpoint, student_config, emulation_config, device, ** _):
        self.frame_dict = defaultdict(lambda: [])
        self.gpu_index = 0
        self.teacher_config = mmcv.Config.fromfile(teacher_config)
        self.student_config = mmcv.Config.fromfile(student_config)
        self.teacher_checkpoint = teacher_checkpoint
        self.student_checkpoint = student_checkpoint
        if emulation_config is not None:
            with open(emulation_config) as f:
                self.emulation_config = json.load(f)
        self.device = device
        self.teacher_model = init_detector(self.teacher_config, self.teacher_checkpoint, device=self.device)
        self.student_dict = {}

    def InitTrainer(self, request, _):
        self.work_dir = request.work_dir
        logging.info(f'Work directory set to {self.work_dir}')
        os.makedirs(os.path.join(self.work_dir, ANNO_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, LABEL_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, FRAME_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, MODEL_DIR), exist_ok=True)
        return trainer_pb2.InitTrainerResponse()

    def SendFrame(self, request, _):
        if self.emulated:
            return trainer_pb2.CloudSendFrameResponse()
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
        self.frame_dict[(request.edge, request.source)] = []
        LOCK.w_release()
        framerate = round(len(indices) / self.emulation_config['retrain_window'] * self.emulation_config['original_framerate'])
        if self.emulated:
            with open(self.emulation_config['profile_path'], 'rb') as f:
                profile = pickle.load(f)[(request.edge, request.source)][:, :, request.version]
            f2c = {}
            for c in self.emulation_config['retrain_cfgs']:
                f2c[self.emulation_config['retrain_cfgs'][c]] = c
            framerate = max(min(framerate, max(f2c)), min(f2c))
            logging.info(f'Estimated framerate {request.source} edge {request.edge} is {framerate} FPS')
            config = f2c[framerate]
            shutil.copyfile(f'{self.emulation_config["model_path"]}/{request.edge}-{request.source}-{request.version}-{config}.pth', self.get_model_dir(request.edge, request.source, request.version))
            # time.sleep(self.emulation_config['training_delay'])
            return self.convert_profile(profile)
        if framerate < self.emulation_config['retrain_cfgs'][1]:
            logging.info(f'Not enough training samples for {request.source} edge {request.edge}, skip training')
            return trainer_pb2.TriggerRetrainResponse(profile=None, updated=False)
        template = self.generate_annotation(request.edge, request.source, indices)
        retrain_dir = self.get_retrain_dir(request.edge, request.source, request.version)
        os.makedirs(retrain_dir, exist_ok=True)
        anno_file = os.path.join(retrain_dir, 'anno.json')
        with open(anno_file, 'w') as f:
            json.dump(template, f)
        cfg = copy.deepcopy(self.student_config)
        cfg.data.train.ann_file = anno_file
        cfg.data.train.img_prefix = ''
        cfg.log_level = 'WARN'
        cfg.work_dir = retrain_dir
        dataset = build_dataset(cfg.data.train)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        train_detector(model, dataset, cfg)
        shutil.move(f'{retrain_dir}/latest.pth', self.get_model_dir(request.edge, request.source, request.version))
        # TODO: fill in retrain profile
        return trainer_pb2.TriggerRetrainResponse(profile=None, updated=True)

    def convert_profile(self, profile):
        p = []
        for line in profile:
            for v in line:
                p.append(v)
        return trainer_pb2.RetProfile(profile=p, updated=True)

    def init_model(self):
        return init_detector(self.student_config, self.student_checkpoint, device=self.device)

    def get_result_dir(self, edge, source, index):
        return os.path.join(self.work_dir, ANNO_DIR, f'{edge}-{source}-{index}.pkl')

    def get_frame_dir(self, edge, source, index):
        return os.path.join(self.work_dir, FRAME_DIR, f'{edge}-{source}-{index}.jpg')

    def get_label_dir(self, edge, source, index):
        return os.path.join(self.work_dir, LABEL_DIR, f'{edge}-{source}-{index}.pkl')

    def get_model_dir(self, edge, source, version):
        return os.path.join(self.work_dir, MODEL_DIR, f'{edge}-{source}-{version}.pth')

    def get_retrain_dir(self, edge, source, version):
        return os.path.join(self.work_dir, RETRAIN_DIR, f'{edge}-{source}-{version}')

    def generate_annotation(self, edge, source, indices, threshold=0.5):
        indices.sort()
        t = copy.deepcopy(TEMPLATE)
        for i, index in enumerate(indices):
            t['images'].append({
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
                    t['annotations'].append({
                        'id': uid,
                        'image_id': i,
                        'category_id': j,
                        'iscrowd': 0,
                        'bbox': converted,
                        'area': converted[2] * converted[3]
                    })
                    uid += 1
        return t

    def xyxy2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
