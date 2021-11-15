import json
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Merge multiple result file')
parser.add_argument(
    '--dir', '-d', help='path to result files', type=str, required=True)
parser.add_argument(
    '--count', '-c', help='number of result files', type=int, required=True)
parser.add_argument(
    '--prefix', '-p', help='prefix of result files', type=str, required=True)
parser.add_argument(
    '--format', '-f', help='format of result files', type=str, default='json')
args = parser.parse_args()

result = []
for i in range(args.count):
    with open(f'{args.dir}/{args.prefix}_{i}.{args.format}', 'rb') as f:
        if args.format == 'pickle':
            o = pickle.load(f)
            result += o
        else:
            o = json.load(f)
            numpy_list = []
            for class_result in o:
                numpy_list.append(np.array(class_result))
            result.append(numpy_list)

with open(f'{args.dir}/{args.prefix}.pkl', 'wb') as f:
    pickle.dump(result, f)
