import os
import json
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(
    description='Move or truncate dataset')
parser.add_argument(
    '--source', '-s', help='source dataset', type=str, required=True)
parser.add_argument(
    '--target', '-t', help='target dataset', type=str, required=True)
parser.add_argument(
    '--begin', '-b', help='begin index of source dataset', type=int, required=True)
parser.add_argument(
    '--end', '-e', help='end index of source dataset (exclusive)', type=int, required=True)
args = parser.parse_args()

os.makedirs(f'data/{args.target}', exist_ok=True)
with open(f'data/annotations/{args.source}.json') as f:
    annotation_source = json.load(f)
with open(f'data/annotations/{args.source}_fake.json') as f:
    annotation_source_fake = json.load(f)
annotation_target = {
    'images': [],
    'categories': annotation_source['categories'],
    'annotations': []
}

annotation_target_fake = {
    'images': [],
    'categories': annotation_source['categories'],
    'annotations': []
}

for i, image in enumerate(annotation_source['images']):
    if image['id'] >= args.begin and image['id'] < args.end:
        image['id'] -= args.begin
        image['file_name'] = f'./{(i - args.begin):06d}.jpg'
        annotation_target['images'].append(image)

for annotation in annotation_source['annotations']:
    if annotation['image_id'] >= args.begin and annotation['image_id'] < args.end:
        annotation['image_id'] -= args.begin
        annotation_target['annotations'].append(annotation)

for i, image in enumerate(annotation_source_fake['images']):
    if image['id'] >= args.begin and image['id'] < args.end:
        image['id'] -= args.begin
        image['file_name'] = f'./{(i - args.begin):06d}.jpg'
        annotation_target_fake['images'].append(image)

for annotation in annotation_source_fake['annotations']:
    if annotation['image_id'] >= args.begin and annotation['image_id'] < args.end:
        annotation['image_id'] -= args.begin
        annotation_target_fake['annotations'].append(annotation)

for i in range(args.begin, args.end):
    copyfile(f'data/{args.source}/{i:06d}.jpg', f'data/{args.target}/{(i - args.begin):06d}.jpg')
with open(f'data/annotations/{args.target}.json', 'w') as f:
    json.dump(annotation_target, f)
with open(f'data/annotations/{args.target}_fake.json', 'w') as f:
    json.dump(annotation_target_fake, f)
