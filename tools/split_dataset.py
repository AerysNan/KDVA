import os
import json
from shutil import copyfile
import argparse


parser = argparse.ArgumentParser(
    description='Generate config file')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, default='data/dataset.json')
parser.add_argument(
    '--size', '-s', help='size of splitted dataset', type=int, default=500)
parser.add_argument(
    '--rate', '-r', help='sampling rate of training dataset', type=int, default=10)
args = parser.parse_args()

postfix = f'{args.size}_{args.rate}'

with open(args.path) as f:
    dataset = json.load(f)
for prefix in dataset:
    k = args.size
    r = args.rate
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
        if offset % r == 0:
            annotation_train_list[epoch]['images'].append(image)
        elif offset % r == (r // 2):
            annotation_val_list[epoch]['images'].append(image)

    for annotation in annotation_golden['annotations']:
        epoch = annotation['image_id'] // k
        offset = annotation['image_id'] % k
        if epoch >= n:
            continue
        if offset % r == 0:
            annotation_train_list[epoch]['annotations'].append(annotation)
        elif offset % r == (r // 2):
            annotation_val_list[epoch]['annotations'].append(annotation)

    for epoch in range(n):
        os.mkdir(f'data/{prefix}_train_{epoch}_{postfix}')
        os.mkdir(f'data/{prefix}_test_{epoch}_{postfix}')
        os.mkdir(f'data/{prefix}_val_{epoch}_{postfix}')
        for i in range(epoch*k, epoch*k+k):
            copyfile(f'data/{prefix}/{i:06}.jpg',
                     f'data/{prefix}_test_{epoch}_{postfix}/{i:06}.jpg')
            if i % r == 0:
                copyfile(f'data/{prefix}/{i:06}.jpg',
                         f'data/{prefix}_train_{epoch}_{postfix}/{i:06}.jpg')
            elif i % r == (r // 2):
                copyfile(f'data/{prefix}/{i:06}.jpg',
                         f'data/{prefix}_val_{epoch}_{postfix}/{i:06}.jpg')
        with open(f'data/annotations/{prefix}_train_{epoch}_{postfix}.json', 'w') as f:
            json.dump(annotation_train_list[epoch], f)
        with open(f'data/annotations/{prefix}_val_{epoch}_{postfix}.json', 'w') as f:
            json.dump(annotation_val_list[epoch], f)
        with open(f'data/annotations/{prefix}_test_{epoch}_{postfix}.json', 'w') as f:
            json.dump(annotation_test_list[epoch], f)
