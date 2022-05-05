import ast
import json
import math
import pickle
import argparse

from evaluate_from_file import evaluate_from_file
from split_dataset import generate_sample_position

ORIGINAL_FRAMERATE = 25


def replay_trace(root, path, name, framerate, size, gt=False, summarize=False, **_):
    with open('datasets.json') as f:
        datasets = json.load(f)
    if not name in datasets:
        key = name[:name.rfind('_')]
    else:
        key = name
    n_epoch = datasets[key]['size'] // size
    if type(framerate) == int:
        framerate = [framerate for _ in range(n_epoch)]
    results = []
    mAP = []
    for i in range(n_epoch):
        with open(f'{root}/{path}/{i:02d}.pkl', 'rb') as f:
            result = pickle.load(f)
        positions = generate_sample_position(framerate[i], ORIGINAL_FRAMERATE)
        for start in range(0, size, ORIGINAL_FRAMERATE):
            for j in range(len(positions) - 1):
                for k in range(positions[j] + 1, positions[j + 1]):
                    result[start + k] = result[start + positions[j]]
            for k in range(positions[-1] + 1, ORIGINAL_FRAMERATE):
                result[start + k] = result[positions[-1]]
        if not summarize:
            mAP.append(evaluate_from_file(result, f'{root}/data/annotations/{key}_test_{i}.{"golden" if not gt else "gt"}.json'))
        results.extend(result)
    mAP.append(evaluate_from_file(results, f'{root}/data/annotations/{key}.{"golden" if not gt else "gt"}.json'))
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replay a single trace')
    parser.add_argument('--root', '-r', help='data root', type=str, required=True)
    parser.add_argument('--path', '-p', help='path to result files', type=str, required=True)
    parser.add_argument('--name', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--framerate', '-f', type=int, default=25, help='replay framerate')
    parser.add_argument('--size', '-s', type=int, required=True, help='chunk size')
    parser.add_argument('--gt', '-g', type=ast.literal_eval, default=False, help='use ground truth or not')
    parser.add_argument('--summarize', type=ast.literal_eval, default=False, help='only evaluate overall mAP')
    args = parser.parse_args()
    mAPs = replay_trace(**args.__dict__)
    classes_of_interest = ['car']
    for v in mAPs:
        mAPs_classwise = [v["classwise"][c] for c in classes_of_interest if not math.isnan(v["classwise"][c])]
        print(f'mAP: {v["bbox_mAP"]} classwise: {sum(mAPs_classwise) / len(mAPs_classwise):.3f}')
