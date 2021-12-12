import argparse
import json

parser = argparse.ArgumentParser(
    description='Convert VIRAT annotation to standard annotation')
parser.add_argument('--base', '-b', type=str, required=True, help='base annotation file')
parser.add_argument('--annotation', '-a', type=str, required=True, help='VIRAT annotation file path')
parser.add_argument('--begin', '-l', type=int, required=True, help='begin index')
parser.add_argument('--end', '-r', type=int, required=True, help='end index')
args = parser.parse_args()

with open(f'data/annotations/{args.base}.base.json') as f:
    data = json.load(f)
results = open(args.annotation)

id = 0
for result in results:
    values = result.split(' ')
    image_id = int(values[2])
    if image_id < args.begin or image_id >= args.end:
        continue
    if len(values) != 8:
        continue
    category = int(values[7])
    if category == 4 or category == 0:
        continue
    annotation = {
        "id": id,
        "image_id": image_id - args.begin,
        "bbox": [int(values[3]), int(values[4]), int(values[5]), int(values[6])],
        "iscrowd": 0,
    }
    annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
    if category == 1:
        annotation["category_id"] = 0
    elif category == 2:
        annotation["category_id"] = 2
    elif category == 3:
        annotation["category_id"] = 7
    elif category == 5:
        annotation["category_id"] = 1
    else:
        print(result)
    data['annotations'].append(annotation)
    id += 1


with open(f'data/annotations/{args.base}.gt.json', 'w') as f:
    json.dump(data, f)
