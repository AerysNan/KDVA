import ast
import shutil
from mmdet.apis import init_detector
import argparse
import pickle
import json
import cv2
import os

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--dataset', '-d', type=str, required=True,
                    help='dataset for detection')
parser.add_argument('--result', '-r', type=str, required=True,
                    help='result file path')
parser.add_argument('--output', '-o', type=str, required=True,
                    help="output name")
parser.add_argument('--video', '-v', type=ast.literal_eval, default=True,
                    help="output video")


parser.add_argument('--begin', '-b', type=int, default=0,
                    help='begin index')
parser.add_argument('--end', '-e', type=int, default=float('inf'),
                    help='end index')
parser.add_argument('--config', '-c', type=str, default='configs/custom/ssd.py',
                    help='configuration path of the model')
parser.add_argument('--model', '-m', type=str, default='checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',
                    help='checkpoint path of the model')

args = parser.parse_args()

model = init_detector(args.config, args.model)

files = os.listdir(f'data/{args.dataset}')
files.sort()

os.makedirs(args.output, exist_ok=True)

if 'pkl' in args.result:
    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    for i, file in enumerate(files):
        if i < args.begin:
            continue
        if i >= args.end:
            break
        print(i)
        image = f'data/{args.dataset}/{file}'
        model.show_result(image, result[i], out_file=f'{args.output}/{i:06d}.jpg', score_thr=0.0)
elif 'json' in args.result:
    with open(args.result) as f:
        result = json.load(f)
    d = {}
    for i in range(args.begin, args.end):
        d[i] = cv2.imread(result['images'][i]['file_name'])
    for annotation in result['annotations']:
        image_id = annotation['image_id']
        if image_id < args.begin:
            continue
        if image_id >= args.end:
            break
        print(image_id)
        bbox = annotation['bbox']
        cv2.rectangle(d[image_id], (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 3)
    for i in range(args.begin, args.end):
        cv2.imwrite(f'{args.output}/{i:06d}.jpg', d[i])

if args.video:
    writer = cv2.VideoWriter(f'{args.output}.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (960, 540))
    for i in range(args.begin, args.end):
        writer.write(cv2.imread(f'{args.output}/{i:06d}.jpg'))
    writer.release()
    shutil.rmtree(f'{args.output}/')
