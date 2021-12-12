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
    '--prefix', '-p', help='prefix of result files', type=str)
parser.add_argument(
    '--format', '-f', help='format of result files', type=str, default='pkl')
args = parser.parse_args()

result = []
for i in range(args.count):
    path = f'{args.dir}/{args.prefix}_{i}.{args.format}' if args.prefix is not None else f'{args.dir}/{i:06d}.{args.format}'
    with open(path, 'rb') as f:
        if args.format == 'pkl':
            o = pickle.load(f)
            if args.prefix is not None:
                result += o
            else:
                result.append(o)
        else:
            o = json.load(f)
            numpy_list = []
            for class_result in o:
                numpy_list.append(np.array(class_result))
            result.append(numpy_list)

with open(f'{args.dir}/result.pkl', 'wb') as f:
    pickle.dump(result, f)
