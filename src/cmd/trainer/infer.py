import os
import json
import copy
import pickle
import threading

import cloud_pb2
from template import template

from mmcv import Config
from mmdet.apis import inference_detector
from mmdet.datasets import build_dataset


class InferThread(threading.Thread):
    def __init__(self, queue, model, interval, config, client):
        super(InferThread, self).__init__()
        self.queue = queue
        self.model = model
        self.interval = interval
        self.config = config
        self.client = client
        self.monitor_dict = {}

    def run(self):
        while True:
            o = self.queue.get()
            edge, source = o['edge'], o['source']
            prefix = f'{edge}_{source}'
            if prefix not in self.monitor_dict:
                self.monitor_dict[prefix] = {
                    'epoch': 0,
                    'interval': 0,
                }
            path = f"dump/data/{prefix}/epoch_{o['epoch']}/{o['index']:06d}.jpg"
            result = inference_detector(self.model, path)
            with open(f"dump/label/{prefix}/epoch_{o['epoch']}/{o['index']:06d}.pkl", 'wb') as f:
                pickle.dump(result, f)
            self.queue.task_done()
            if o['index'] // self.interval > self.monitor_dict[prefix]['interval']:
                monitor_thread = MonitorThread(
                    edge=edge,
                    source=source,
                    epoch=self.monitor_dict[prefix]['epoch'],
                    begin=self.monitor_dict[prefix]['interval'] * self.interval,
                    end=self.monitor_dict[prefix]['interval'] * self.interval + self.interval,
                    config=self.config,
                    client=self.client
                )
                self.monitor_dict[prefix]['interval'] += 1
                self.monitor_dict[prefix]['epoch'] = o['epoch']
                monitor_thread.start()


class MonitorThread(threading.Thread):
    def __init__(self, edge, source, epoch, begin, end, config, client):
        super(MonitorThread, self).__init__()
        self.edge = edge
        self.source = source
        self.epoch = epoch
        self.begin = begin
        self.end = end
        self.config = config
        self.client = client

    def run(self):
        prefix = f'{self.edge}_{self.source}'
        # add categories
        d = copy.deepcopy(template)
        # add images
        files = os.listdir(f'dump/data/{prefix}/epoch_{self.epoch}')
        files.sort()
        for i, name in enumerate(files):
            filename, _ = os.path.splitext(name)
            if not int(filename) >= self.begin or not int(filename) < self.end:
                continue
            d['images'].append({
                'id': i,
                'file_name': name,
                'height': 1080,
                'width': 1920
            })
        # add annotations
        files = os.listdir(f'dump/label/{prefix}/epoch_{self.epoch}')
        files.sort()
        id = 0
        for i, name in enumerate(files):
            filename, _ = os.path.splitext(name)
            if not int(filename) >= self.begin or not int(filename) < self.end:
                continue
            with open(f'dump/label/{prefix}/epoch_{self.epoch}/{name}', 'rb') as f:
                result = pickle.load(f)
            for label in range(len(result)):
                bboxes = result[label]
                for bbox in bboxes:
                    if bbox[4] < 0.8:
                        continue
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
        with open(f'dump/label/{prefix}/monitor_{self.begin}_{self.end}.json', 'w') as f:
            json.dump(d, f)
        # generate result
        result = []
        files = os.listdir(f'dump/fake/{prefix}/epoch_{self.epoch}')
        files.sort()
        for name in files:
            filename, _ = os.path.splitext(name)
            if not int(filename) >= self.begin or not int(filename) < self.end:
                continue
            with open(f'dump/fake/{prefix}/epoch_{self.epoch}/{name}', 'rb') as f:
                result.append(pickle.load(f))
        if len(result) == 0:
            print(f'#####{self.source}#####{self.epoch}#####{self.begin}#####')
        # generate configuration
        cfg = Config.fromfile(self.config)
        cfg.data.test.ann_file = f'dump/label/{prefix}/monitor_{self.begin}_{self.end}.json'
        cfg.data.test.img_prefix = f'dump/data/{prefix}/epoch_{self.epoch}'
        dataset = build_dataset(cfg.data.test)
        evaluation = dataset.evaluate(result, metric='bbox')
        self.client.ReportProfile(cloud_pb2.ReportProfileRequest(
            edge=self.edge,
            source=self.source,
            begin=self.begin,
            end=self.end,
            accuracy=evaluation['bbox_mAP']
        ))

    def xyxy2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
