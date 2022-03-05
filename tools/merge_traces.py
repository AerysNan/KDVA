import ast
from shutil import copyfile
import os
import json
import argparse


parser = argparse.ArgumentParser(description='Merge traces')
parser.add_argument('--output', '-o', type=str, required=True, help='Output trace name')
parser.add_argument('--input', '-i', type=str, required=True, help='Input traces file')
parser.add_argument('--gt', '-g', type=ast.literal_eval, default=True, help='Generate groundtruth file')

args = parser.parse_args()

image_count, annotation_id, image_id = 0, 0, 0

output_annotation = {
    'images': [],
    'annotations': [],
    'categories': [],
    'ignored_regions': []
}

datasets = []

f = open(args.input)
for line in f:
    datasets.append(line[:-1])
f.close()

annotation_only = False

if not annotation_only:
    os.makedirs(f'data/{args.output}', exist_ok=True)


for i, dataset in enumerate(datasets):
    d = {}
    with open(f'data/annotations/{dataset}.{"gt" if args.gt else "golden"}.json') as f:
        dataset_annotation = json.load(f)
    if not annotation_only:
        files = os.listdir(f'data/{dataset}')
        files.sort()
        for file in files:
            copyfile(f'data/{dataset}/{file}', f'data/{args.output}/{image_count:06d}.jpg')
            image_count += 1
    else:
        image_count += len(dataset_annotation['images'])
    for j, image in enumerate(dataset_annotation['images']):
        d[image['id']] = image_id
        image['id'] = image_id
        image_id += 1
        if not annotation_only:
            image['file_name'] = f'./data/{args.output}/{image["id"]:06d}.jpg'
        output_annotation['images'].append(image)
    for annotation in dataset_annotation['annotations']:
        annotation['image_id'] = d[annotation['image_id']]
        annotation['id'] = annotation_id
        annotation_id += 1
        output_annotation['annotations'].append(annotation)
    output_annotation['categories'] = dataset_annotation['categories']
    if 'ignored_regions' in dataset_annotation:
        for region in dataset_annotation['ignored_regions']:
            region['begin'] = d[region['begin']]
            region['end'] = d[region['end'] - 1] + 1
            output_annotation['ignored_regions'].append(region)

with open(f'data/annotations/{args.output}.{"gt" if args.gt else "golden"}.json', 'w') as f:
    json.dump(output_annotation, f)
