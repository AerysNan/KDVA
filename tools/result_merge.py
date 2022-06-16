import os
import pickle
import argparse


def merge_result(input, output, **_):
    result = []
    files = os.listdir(input)
    files.sort()
    for file in files:
        if not 'pkl' in file:
            continue
        with open(os.path.join(input, file), 'rb') as f:
            o = pickle.load(f)
            result.extend(o)

    with open(output, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge multiple result file')
    parser.add_argument('--input', '-i', help='path to input result files', type=str, required=True)
    parser.add_argument('--output', '-o', help='output result file', type=str, required=True)
    args = parser.parse_args()
    merge_result(**args.__dict__)
