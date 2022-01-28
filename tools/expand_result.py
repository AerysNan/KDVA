import os
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Expand result file to individual frame files')
parser.add_argument(
    '--result', '-r', help='result file', type=str, required=True)
parser.add_argument(
    '--dest', '-d', help='target directory', type=str, required=True)
parser.add_argument(
    '--size', '-s', help='chunk size', type=int, default=500)
args = parser.parse_args()

os.makedirs(f'{args.dest}', exist_ok=True)

with open(f'{args.result}', 'rb') as f:
    results = pickle.load(f)
for i in range(0, len(results), args.size):
    with open(f'{args.dest}/{i // args.size:02d}.pkl', 'wb') as f:
        pickle.dump(results[i: i + args.size], f)
    print(i)
