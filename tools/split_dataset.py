import os
import json
from shutil import copyfile
import argparse


parser = argparse.ArgumentParser(
    description='Generate config file')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, required=True)
args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)
for prefix in dataset:
    k = 500
    n = dataset[prefix] // k

    annotation_all = json.load(open(f'data/annotations/{prefix}.json'))
    annotation_golden = json.load(open(f'data/annotations/{prefix}_fake.json'))

    annotation_train_list = [
        {
            'images': [],
            'annotations':[],
            'categories':annotation_all['categories']
        } for _ in range(n)
    ]

    annotation_val_list = [
        {
            'images': [],
            'annotations':[],
            'categories':annotation_all['categories']
        } for _ in range(n)
    ]

    annotation_test_list = [
        {
            'images': [],
            'annotations':[],
            'categories':annotation_all['categories']
        } for _ in range(n)
    ]

    for image in annotation_all['images']:
        epoch = image['id'] // k
        if epoch >= n:
            continue
        annotation_test_list[epoch]['images'].append(image)

    for annotation in annotation_all['annotations']:
        epoch = annotation['image_id'] // k
        if epoch >= n:
            continue
        annotation_test_list[epoch]['annotations'].append(annotation)

    for image in annotation_golden['images']:
        epoch = image['id'] // k
        offset = image['id'] % k
        if epoch >= n:
            continue
        if offset % 10 == 0:
            annotation_train_list[epoch]['images'].append(image)
        elif offset % 50 == 25:
            annotation_val_list[epoch]['images'].append(image)

    for annotation in annotation_golden['annotations']:
        epoch = annotation['image_id'] // k
        offset = annotation['image_id'] % k
        if epoch >= n:
            continue
        if offset % 10 == 0:
            annotation_train_list[epoch]['annotations'].append(annotation)
        elif offset % 50 == 25:
            annotation_val_list[epoch]['annotations'].append(annotation)

    for epoch in range(n):
        os.mkdir(f'data/{prefix}_train_{epoch}')
        os.mkdir(f'data/{prefix}_test_{epoch}')
        os.mkdir(f'data/{prefix}_val_{epoch}')
        for i in range(epoch*k, epoch*k+k):
            copyfile(f'data/{prefix}/{i:06}.jpg',
                     f'data/{prefix}_test_{epoch}/{i:06}.jpg')
            if i % 10 == 0:
                copyfile(f'data/{prefix}/{i:06}.jpg',
                         f'data/{prefix}_train_{epoch}/{i:06}.jpg')
            elif i % 50 == 25:
                copyfile(f'data/{prefix}/{i:06}.jpg',
                         f'data/{prefix}_val_{epoch}/{i:06}.jpg')
        with open(f'data/annotations/{prefix}_train_{epoch}.json', 'w') as f:
            json.dump(annotation_train_list[epoch], f)
        with open(f'data/annotations/{prefix}_val_{epoch}.json', 'w') as f:
            json.dump(annotation_val_list[epoch], f)
        with open(f'data/annotations/{prefix}_test_{epoch}.json', 'w') as f:
            json.dump(annotation_test_list[epoch], f)
