import argparse
import json

parser = argparse.ArgumentParser(
    description='Convert the result from a model to grouth truth annotations')
parser.add_argument('--base', '-b', type=str, required=True,
                    help='base annotation file')
parser.add_argument('--result', '-r', type=str, required=True,
                    help='inference result from a model')
parser.add_argument('--threshold', '-t', type=float, default=0.8,
                    help='confidence threshold for result filter')
args = parser.parse_args()

with open(f'data/annotations/{args.base}_base.json') as f:
    data = json.load(f)

results = json.load(open(args.result))

id = 0
for result in results:
    if result['score'] < args.threshold:
        continue
    del result['score']
    result['id'] = id
    result['area'] = result['bbox'][2] * result['bbox'][3]
    result['iscrowd'] = 0
    data['annotations'].append(result)
    id += 1

with open(f'data/annotations/{args.base}_fake.json', 'w') as f:
    json.dump(data, f)
