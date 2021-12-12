import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(
    description='Move or truncate dataset')
parser.add_argument(
    '--source', '-s', help='source dataset', type=str, required=True)
parser.add_argument(
    '--target', '-t', help='target dataset', type=str, required=True)
parser.add_argument(
    '--begin', '-b', help='begin index of source dataset', type=int, required=True)
parser.add_argument(
    '--end', '-e', help='end index of source dataset (exclusive)', type=int, required=True)
args = parser.parse_args()

os.makedirs(f'data/{args.target}', exist_ok=True)

for i in range(args.begin, args.end):
    copyfile(f'data/datasets/{args.source}/{i:06d}.png', f'data/{args.target}/{(i - args.begin):06d}.jpg')
