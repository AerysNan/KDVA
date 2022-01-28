import os
import json
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Merge multiple result file')
parser.add_argument(
    '--dir', '-d', help='path to result files', type=str, required=True)
parser.add_argument(
    '--output', '-o', help='output directory', type=str, required=True)
args = parser.parse_args()

result = []
files = os.listdir(args.dir)
files.sort()
for file in files:
    if not 'pkl' in file:
        continue
    with open(f'{args.dir}/{file}', 'rb') as f:
        o = pickle.load(f)
        result.extend(o)

with open(f'{args.output}', 'wb') as f:
    pickle.dump(result, f)
