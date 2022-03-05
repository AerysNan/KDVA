import json
import pickle
import argparse

from evaluate_from_file import evaluate_from_file
ORIGINAL_FRAMERATE = 500


def generate_sample_position(sample_count, sample_interval):
    sample_win, total = [1 for _ in range(sample_count)], sample_count
    while total < sample_interval:
        for i in range(sample_count):
            sample_win[i] += 1
            total += 1
            if total == sample_interval:
                break
    pos = [0]
    for i in range(sample_count - 1):
        pos.append(pos[-1] + sample_win[i])
    return pos


def replay_trace(path, name, framerate, batch_size):
    with open('datasets.json') as f:
        datasets = json.load(f)
    if not name in datasets:
        key = name[:name.rfind('_')]
    else:
        key = name
    n_epoch = datasets[key]['size'] // batch_size
    if type(framerate) == int:
        framerate = [framerate for _ in range(n_epoch)]
    results = []
    mAP = []
    for i in range(n_epoch):
        with open(f'{path}/{i:02d}.pkl', 'rb') as f:
            result = pickle.load(f)
        positions = generate_sample_position(framerate[i], ORIGINAL_FRAMERATE)
        for j in range(len(positions) - 1):
            for k in range(positions[j] + 1, positions[j + 1]):
                result[k] = result[positions[j]]
        for k in range(positions[-1] + 1, len(result)):
            result[k] = result[positions[-1]]
        mAP.append(evaluate_from_file(result, f'data/annotations/{name}_test_{i}.gt.json'))
        results.extend(result)

    mAP.append(evaluate_from_file(results, f'data/annotations/{key}.gt.json'))
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replay a single trace')
    parser.add_argument('--path', '-p', help='path to result files', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--framerate', '-f', type=int, default=60, help='replay framerate')
    parser.add_argument('--size', '-s', type=int, required=True, help='chunk size')
    args = parser.parse_args()
    mAPs = replay_trace(args.path, args.dataset, args.framerate, args.size)
    for mAP in mAPs:
        print(mAP['bbox_mAP'])
    print('classwise')
    for mAP in mAPs:
        print(mAP["bbox_mAP_car"])
