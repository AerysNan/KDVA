import os
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Expand result file to individual frame files')
parser.add_argument(
    '--result', '-r', help='result file', type=str, required=True)
parser.add_argument(
    '--dest', '-d', help='target directory', type=str, required=True)
args = parser.parse_args()

os.makedirs(f'{args.dest}', exist_ok=True)
with open(f'{args.result}', 'rb') as f:
    results = pickle.load(f)
for i, result in enumerate(results):
    with open(f'{args.dest}/{i:06d}.pkl', 'wb') as f:
        pickle.dump(result, f)
    print(i)
