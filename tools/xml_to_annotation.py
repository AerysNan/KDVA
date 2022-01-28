import json
import argparse
import xmltodict

parser = argparse.ArgumentParser(description='Convert XML to annotation file')
parser.add_argument('--path', '-p', help='XML file path', type=str, required=True)
parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
args = parser.parse_args()

with open(args.path) as f:
    content = f.read()

parsed_content = xmltodict.parse(content)['sequence']
if parsed_content['ignored_region'] is None:
    ignored_regions = []
else:
    ignored_regions = parsed_content['ignored_region']['box']
annotations = parsed_content['frame']

with open(f'data/annotations/{args.dataset}.base.json') as f:
    data = json.load(f)
uid = 0
data['ignored_regions'] = []


def parse_annotation(target, num):
    category = target['attribute']['@vehicle_type']
    annotation = {
        'id': uid,
        'image_id': num,
        'bbox': [float(target['box']['@left']), float(target['box']['@top']), float(target['box']['@width']), float(target['box']['@height'])],
        'iscrowd': 0,
    }
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    if category == 'car':
        annotation['category_id'] = 2
    elif category == 'others' or category == 'van':
        annotation['category_id'] = 7
    elif category == 'bus':
        annotation['category_id'] = 5
    return annotation


for annotation in annotations:
    num = int(annotation['@num']) - 1
    if type(annotation['target_list']['target']) == list:
        for target in annotation['target_list']['target']:
            data['annotations'].append(parse_annotation(target, num))
            uid += 1
    else:
        data['annotations'].append(parse_annotation(annotation['target_list']['target'], num))
        uid += 1
if type(ignored_regions) == list:
    for region in ignored_regions:
        data['ignored_regions'].append({
            "begin": 0,
            "end": len(data['images']),
            "region": [float(region['@left']), float(region['@top']), float(region['@left']) + float(region['@width']), float(region['@top']) + float(region['@height'])]
        })
else:
    data['ignored_regions'].append({
        "begin": 0,
        "end": len(data['images']),
        "region": [float(ignored_regions['@left']), float(ignored_regions['@top']), float(ignored_regions['@left']) + float(ignored_regions['@width']), float(ignored_regions['@top']) + float(ignored_regions['@height'])]
    })

with open(f'data/annotations/{args.dataset}.gt.json', 'w') as f:
    json.dump(data, f)
