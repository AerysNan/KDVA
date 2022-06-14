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
for i in range(args.begin, args.end):
    copyfile(f'data/{args.source}/{i:06d}.jpg', f'data/{args.target}/{(i - args.begin):06d}.jpg')

with open(f'data/annotations/{args.source}.gt.json') as f:
    source_annotations = json.load(f)

target_annotations = {
    'images': [],
    'annotations': [],
    'categories': source_annotations['categories'],
}

for image in source_annotations['images']:
    if image['id'] < args.begin or image['id'] >= args.end:
        continue
    image['id'] -= args.begin
    image['file_name'] = f'./data/{args.target}/{image["id"]:06d}.jpg'
    target_annotations['images'].append(image)
for annotation in source_annotations['annotations']:
    if annotation['image_id'] < args.begin or annotation['image_id'] >= args.end:
        continue
    annotation['image_id'] -= args.begin
    target_annotations['annotations'].append(annotation)
if "ignored_regions" in source_annotations:
    target_annotations['ignored_regions'] = []
    for region in source_annotations['ignored_regions']:
        if region['begin'] >= args.end or region['end'] <= args.begin:
            continue
        region['begin'] = max(region['begin'], args.begin) - args.begin
        region['end'] = min(region['end'], args.end) - args.begin
        target_annotations['ignored_regions'].append(region)
with open(f'data/annotations/{args.target}.gt.json', 'w') as f:
    json.dump(target_annotations, f)
