import os
import json
import mmcv
import copy
import pickle
import threading

import cloud_pb2

from template import template

from mmdet.apis import inference_detector, train_detector
from mmdet.datasets.builder import build_dataset
from mmdet.models import build_detector


class InferThread(threading.Thread):
    def __init__(self, queue, model):
        super(InferThread, self).__init__()
        self.queue = queue
        self.model = model

    def run(self):
        while True:
            path = self.queue.get()
            result = inference_detector(self.model, path)
            with open(path.replace('data', 'label').replace('jpg', 'pkl'), 'wb') as f:
                pickle.dump(result, f)
            self.queue.task_done()


class DistillThread(threading.Thread):
    def __init__(self, client, prefix, epoch):
        super(DistillThread, self).__init__()
        self.client = client
        self.prefix = prefix
        self.epoch = epoch

    def run(self):
        print(f'Start distillation {self.prefix} on epoch {self.epoch}')
        config_file = 'configs/custom/ssd.py'
        cfg = mmcv.Config.fromfile(config_file)
        self.generate_annotation()
        cfg.data.train.ann_file = f'dump/label/{self.prefix}/epoch_{self.epoch}.json'
        cfg.data.train.img_prefix = f'dump/data/{self.prefix}/epoch_{self.epoch}'
        cfg.log_level = 'WARN'
        cfg.work_dir = f'dump/distill/{self.prefix}/epoch_{self.epoch}'
        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.gpu_ids = [1]
        cfg.seed = None
        dataset = build_dataset(cfg.data.train)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        train_detector(model, dataset, cfg)
        print(f'Finish distillation {self.prefix} on epoch {self.epoch}')
        with open(f'dump/distill/{self.prefix}/epoch_{self.epoch}/latest.pth', 'rb') as f:
            self.client.DeliverModel(cloud_pb2.DeliverModelRequest(
                prefix=self.prefix, epoch=self.epoch, model=f.read()))

    def generate_annotation(self):
        d = copy.deepcopy(template)
        files = os.listdir(f'dump/data/{self.prefix}/epoch_{self.epoch}')
        files.sort()
        for i, name in enumerate(files):
            d['images'].append({
                'id': i,
                'file_name': name,
                'height': 1080,
                'width': 1920
            })
        files = os.listdir(f'dump/label/{self.prefix}/epoch_{self.epoch}')
        files.sort()
        id = 0
        for i, name in enumerate(files):
            with open(f'dump/label/{self.prefix}/epoch_{self.epoch}/{name}', 'rb') as f:
                result = pickle.load(f)
            for label in range(len(result)):
                bboxes = result[label]
                for bbox in bboxes:
                    converted = self.xyxy2xywh(bbox.tolist())
                    d['annotations'].append({
                        'id': id,
                        'image_id': i,
                        'category_id': label,
                        'iscrowd': 0,
                        'bbox': converted,
                        'area': converted[2]*converted[3]
                    })
                    id += 1
        with open(f'dump/label/{self.prefix}/epoch_{self.epoch}.json', 'w') as f:
            json.dump(d, f)

    def xyxy2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
