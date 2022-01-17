from shutil import copyfile
import os
import json
import argparse


parser = argparse.ArgumentParser(description='Merge traces')
parser.add_argument('--output', '-o', type=str, required=True, help='Output trace name')
parser.add_argument('--input', '-i', type=str, required=True, help='Input traces file')

args = parser.parse_args()

os.makedirs(f'data/{args.output}', exist_ok=True)
image_count = 0
dataset_offset = []
global_id = 0

output_annotation = {
    'images': [],
    'annotations': [],
    'categories': [],
}

datasets = []

f = open(args.input)
for line in f:
    datasets.append(line[:-1])
f.close()


for i, dataset in enumerate(datasets):
    dataset_offset.append(image_count)
    files = os.listdir(f'data/{dataset}')
    files.sort()
    for file in files:
        copyfile(f'data/{dataset}/{file}', f'data/{args.output}/{image_count:06d}.jpg')
        image_count += 1
    with open(f'data/annotations/{dataset}.gt.json') as f:
        dataset_annotation = json.load(f)
    for j, image in enumerate(dataset_annotation['images']):
        id = image['id']
        image['id'] = id + dataset_offset[-1]
        image['file_name'] = f'./data/{args.output}/{image["id"]:06d}.jpg'
        output_annotation['images'].append(image)
    for annotation in dataset_annotation['annotations']:
        annotation['image_id'] += dataset_offset[-1]
        annotation['id'] = global_id
        global_id += 1
        output_annotation['annotations'].append(annotation)
    output_annotation['categories'] = dataset_annotation['categories']

with open(f'data/annotations/{args.output}.gt.json', 'w') as f:
    json.dump(output_annotation, f)
