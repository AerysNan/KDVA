from mmdet.apis import init_detector, inference_detector
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--dataset', '-d', type=str, required=True,
                    help='dataset for detection')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model use for detection')
parser.add_argument('--gpu', '-g', type=str, default='cuda',
                    help="name of GPU device to run inference")
args = parser.parse_args()

models = json.load(open('model.json'))

if not args.model in models:
    print('Please specify a model in ./model.json')
    sys.exit(1)

config_file = f"{os.getcwd()}/configs/{models[args.model]['config']}"
checkpoint_file = f"{os.getcwd()}/checkpoints/{models[args.model]['checkpoint']}"
model = init_detector(config_file, checkpoint_file, device=args.gpu)

if not os.path.exists(f'data/{args.dataset}'):
    print('Please specify a dataset in ./data/')
    sys.exit(1)

image = f'data/{args.dataset}/000000.jpg'
result = inference_detector(model, image)
model.show_result(image, result, out_file='result.jpg', score_thr=0.2)
