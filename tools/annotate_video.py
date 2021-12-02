from mmdet.apis import init_detector
import argparse
import pickle
import json
import time
import sys
import os

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--dataset', '-d', type=str, required=True,
                    help='dataset for detection')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model use for detection')
parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                    help="name of GPU device to run inference")
args = parser.parse_args()

with open('data/model.json') as f:
    models = json.load(f)
with open('data/dataset.json') as f:
    datasets = json.load(f)

if not args.model in models:
    print('Please specify a model in data/model.json')
    sys.exit(1)

if not args.dataset in datasets:
    print('Please specify a dataset in data/dataset.json')
    sys.exit(1)

config_file = f"{os.getcwd()}/configs/{models[args.model]['config']}"
checkpoint_file = f"{os.getcwd()}/checkpoints/{models[args.model]['checkpoint']}"
model = init_detector(config_file, checkpoint_file, device=args.gpu)
os.makedirs(f'video/{args.dataset}', exist_ok=True)

with open(f'data/{args.dataset}_{args.model}.pkl', 'rb') as f:
    result = pickle.load(f)

for i in range(datasets[args.dataset]):
    image = f'data/{args.dataset}/{i:06d}.jpg'
    model.show_result(image, result[i], out_file=f'video/{args.dataset}/{i:06d}.jpg')
    print(i)
