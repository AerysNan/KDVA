import json
import argparse


parser = argparse.ArgumentParser(
    description='Generate config file')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, default='datasets.json')
parser.add_argument(
    '--size', '-s', help='size of splitted dataset', type=int, default=600)
parser.add_argument(
    '--rate', '-r', help='sampling rate of training dataset', type=int, default=10)
parser.add_argument(
    '--postfix', '-o', help='generated postfix', type=str)
args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)
if args.postfix:
    postfix = f'_{args.postfix}'
else:
    postfix = ''
for prefix in dataset:
    k = args.size
    r = args.rate
    n = dataset[prefix]['size'] // k

    annotation_all = json.load(open(f'data/annotations/{prefix}.gt.json'))

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
        offset = image['id'] % k
        annotation_test_list[epoch]['images'].append(image)
        if offset % r == 0:
            annotation_train_list[epoch]['images'].append(image)
        elif offset % r == (r // 2):
            annotation_val_list[epoch]['images'].append(image)

    for annotation in annotation_all['annotations']:
        epoch = annotation['image_id'] // k
        offset = annotation['image_id'] % k
        annotation_test_list[epoch]['annotations'].append(annotation)
        if offset % r == 0:
            annotation_train_list[epoch]['annotations'].append(annotation)
        elif offset % r == (r // 2):
            annotation_val_list[epoch]['annotations'].append(annotation)

    for epoch in range(n):

        with open(f'data/annotations/{prefix}{postfix}_train_{epoch}.gt.json', 'w') as f:
            json.dump(annotation_train_list[epoch], f)
        with open(f'data/annotations/{prefix}{postfix}_test_{epoch}.gt.json', 'w') as f:
            json.dump(annotation_test_list[epoch], f)
        with open(f'data/annotations/{prefix}{postfix}_val_{epoch}.gt.json', 'w') as f:
            json.dump(annotation_val_list[epoch], f)
