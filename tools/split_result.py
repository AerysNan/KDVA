import os
import pickle
import argparse


parser = argparse.ArgumentParser(description="Split result")
parser.add_argument(
    "--path", "-p", help="path to result file", type=str, required=True
)
parser.add_argument(
    "--size", "-s", help="size of splitted result", type=int, default=500
)
parser.add_argument("--output", "-o", help="output name", type=str, required=True)
args = parser.parse_args()

with open(args.path, 'rb') as f:
    results = pickle.load(f)

os.makedirs(args.output, exist_ok=True)

for i in range(len(results) // args.size):
    with open(f'{args.output}/{i:02d}.pkl', 'wb') as f:
        pickle.dump(results[i * args.size: (i * args.size + args.size)], f)
