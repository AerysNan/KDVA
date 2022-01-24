import os
import cv2
import ast
import json
import shutil
import pickle
import argparse
from evaluate_from_file import filter_result

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

args = parser.parse_args()

files = os.listdir(f'data/{args.dataset}')
files.sort()

color = {
    2: (0, 0, 255),
    5: (0, 255, 0),
    7: (255, 0, 0)
}

os.makedirs(args.output, exist_ok=True)

if 'pkl' in args.result:
    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    with open(f'data/annotations/{args.dataset}.gt.json') as f:
        gt = json.load(f)
    if "ignored_regions" in gt:
        print('Ignored regions detected, start filtering...')
        filter_result(result, gt['ignored_regions'])
        print('Filtering finished!')
    for i in range(args.begin, args.end):
        print(i)
        img = cv2.imread(f'data/{args.dataset}/{files[i]}')
        for j, class_result in enumerate(result[i]):
            if j not in color:
                continue
            for bbox in class_result:
                if bbox[4] >= 0.3:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color[j], 3)
        cv2.imwrite(f'{args.output}/{i:06d}.jpg', img)
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
