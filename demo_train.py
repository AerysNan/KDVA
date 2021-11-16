import mmcv
from mmdet.apis import train_detector
from mmdet.datasets.builder import build_dataset
from mmdet.models import build_detector
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model use for detection')
parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                    help="name of GPU device to run inference")
args = parser.parse_args()

with open('data/model.json') as f:
    models = json.load(f)

if not args.model in models:
    print('Please specify a model in data/model.json')
    sys.exit(1)

config_file = f"{os.getcwd()}/configs/{models[args.model]['config']}"

cfg = mmcv.Config.fromfile(config_file)
cfg.work_dir = 'tmp/'
cfg.gpu_ids = [0]
cfg.seed = None
dataset = build_dataset(cfg.data.train)
model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

train_detector(model, dataset, cfg)
