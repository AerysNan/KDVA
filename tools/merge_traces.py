import os
import ast
import json
import argparse


def merge_traces(input_file, output_file):
    image_count, annotation_id, image_id = 0, 0, 0
    output_annotation = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    datasets = []
    with open(input_file) as f:
        for line in f:
            datasets.append(line[:-1])

    for dataset in datasets:
        id2id = {}
        with open(dataset) as f:
            dataset_annotation = json.load(f)
        image_count += len(dataset_annotation['images'])
        for image in dataset_annotation['images']:
            id2id[image['id']] = image_id
            image['id'] = image_id
            image_id += 1
            output_annotation['images'].append(image)
        for annotation in dataset_annotation['annotations']:
            annotation['image_id'] = id2id[annotation['image_id']]
            annotation['id'] = annotation_id
            annotation_id += 1
            output_annotation['annotations'].append(annotation)
        output_annotation['categories'] = dataset_annotation['categories']

    with open(f'{args.root}/data/annotations/{args.output}.{"gt" if args.gt else "golden"}.json', 'w') as f:
        json.dump(output_annotation, f)


parser = argparse.ArgumentParser(description='Merge traces')
parser.add_argument('--root', '-r', type=str, required=True, help='Data root')
parser.add_argument('--output', '-o', type=str, required=True, help='Output trace name')
parser.add_argument('--input', '-i', type=str, required=True, help='Input traces file')
parser.add_argument('--gt', '-g', type=ast.literal_eval, default=True, help='Generate groundtruth file')
parser.add_argument('--annotation-only', '-a', type=ast.literal_eval, default=False, help='Only generate annotation file for merged dataset')

# args = parser.parse_args()
