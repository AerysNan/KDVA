import argparse
import json

parser = argparse.ArgumentParser(
    description='Convert the result from a model to grouth truth annotations')
parser.add_argument('--base', '-b', type=str, required=True,
                    help='base annotation file')
parser.add_argument('--result', '-r', type=str, required=True,
                    help='inference result from a model')
args = parser.parse_args()

with open(f'data/annotations/{args.base}_fake.json') as f:
    data = json.load(f)
results = open(args.result)

id = 0
for result in results:
    values = result.split(' ')
    if len(values) != 8:
        continue
    category = int(values[7])
    if category == 4 or category == 0:
        continue
    annotation = {
        "id": id,
        "image_id": int(values[2]),
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


with open(f'data/annotations/{args.base}.json', 'w') as f:
    json.dump(data, f)
