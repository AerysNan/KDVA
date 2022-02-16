import os
import pickle
import argparse


def merge_result(dir, output):
    result = []
    files = os.listdir(dir)
    files.sort()
    for file in files:
        if not 'pkl' in file:
            continue
        with open(f'{dir}/{file}', 'rb') as f:
            o = pickle.load(f)
            result.extend(o)

    with open(f'{output}', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge multiple result file')
    parser.add_argument(
        '--dir', '-d', help='path to result files', type=str, required=True)
    parser.add_argument(
        '--output', '-o', help='output directory', type=str, required=True)
    args = parser.parse_args()
    merge_result(args.dir, args.output)
