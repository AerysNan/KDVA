import json
import argparse
from xmldict import xml_to_dict

parser = argparse.ArgumentParser(description='Convert XML to annotation file')
parser.add_argument('--path', '-p', help='XML file path', type=str, required=True)
parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
args = parser.parse_args()

with open(args.path) as f:
    content = f.read()

annotations = xml_to_dict(content)['sequence']['frame']

with open(f'data/annotations/{args.dataset}.base.json') as f:
    data = json.load(f)
uid = 0


def parse_annotation(target):
    category = target['attribute']['@vehicle_type']
    annotation = {
        'id': uid,
        'image_id': i,
        'bbox': [float(target['box']['@left']), float(target['box']['@top']), float(target['box']['@width']), float(target['box']['@height'])],
        'iscrowd': 0,
    }
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    if category == 'car':
        annotation['category_id'] = 2
    elif category == 'van' or category == 'others':
        annotation['category_id'] = 7
    elif category == 'bus':
        annotation['category_id'] = 5
    return annotation


for i, annotation in enumerate(annotations):
    if type(annotation['target_list']['target']) == list:
        for target in annotation['target_list']['target']:
            data['annotations'].append(parse_annotation(target))
            uid += 1
    else:
        data['annotations'].append(parse_annotation(annotation['target_list']['target']))
        uid += 1


with open(f'data/annotations/{args.dataset}.gt.json', 'w') as f:
    json.dump(data, f)
