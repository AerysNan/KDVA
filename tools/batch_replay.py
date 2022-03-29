from multiprocessing import Pool

from replay_trace import replay_trace
from evaluate_from_file import evaluate_from_file
import numpy as np
import argparse
import pickle
import math


def batch_replay(begin, end, count, prefix, batch_size=500, **_):
    m, m_class = np.zeros((7, 5, 12, 13), dtype=np.double), np.zeros((7, 5, 12, 13), dtype=np.double)
    p, output = Pool(processes=10), {}

    for retrain in range(7):
        for f in range(5, 30, 5):
            for stream in range(1, 13):
                output[(retrain, f, stream)] = p.apply_async(replay_trace, (
                    f'snapshot/result/detrac_{stream}_{retrain}_e40' if retrain > 0 else f'snapshot/result/detrac_{stream}_{retrain}',
                    f'detrac_{stream}', f, 500,
                ))

    # for retrain in range(7):
    #     for f in range(5, 30, 5):
    #         for stream in range(1, 13):
    #             for epoch in range(10):
    #                 output[(retrain, f, stream, epoch)] = p.apply_async(evaluate_from_file, (
    #                     f'snapshot/result/down_{stream}_{retrain}_vna/{epoch:02d}.pkl' if retrain > 0 else f'snapshot/result/down_{stream}_{retrain}/{epoch:02d}.pkl',
    #                     f'data/annotations/down_{stream}_test_{epoch}.golden.json', (f, 25),
    #                 ))

    p.close()
    p.join()

    for retrain in range(7):
        for f in range(5, 30, 5):
            for stream in range(1, 13):
                result = output[(retrain, f, stream)].get()
                for epoch in range(len(result)):
                    m[retrain, (f // 5) - 1, stream - 1, epoch] = result[epoch]["bbox_mAP"]
                    classes_of_interest = ['car']
                    mAPs_classwise = [result[epoch]["classwise"][c] for c in classes_of_interest if not math.isnan(result[epoch]["classwise"][c])]
                    m_class[retrain, (f // 5) - 1, stream - 1, epoch] = sum(mAPs_classwise) / len(mAPs_classwise)

    # for retrain in range(7):
    #     for f in range(5, 30, 5):
    #         for stream in range(1, 13):
    #             for epoch in range(10):
    #                 result = output[(retrain, f, stream, epoch)].get()
    #                 m[retrain, (f // 5) - 1, stream - 1, epoch + 2] = result["bbox_mAP"]
    #                 classes_of_interest = ['car']
    #                 mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
    #                 m_class[retrain, (f // 5) - 1, stream - 1, epoch + 2] = sum(mAPs_classwise) / len(mAPs_classwise)

    with open(f'{prefix}_replay_{begin}_{end}.pkl', 'wb') as f:
        pickle.dump({"data": m, "classwise_data": m_class}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate validation result")
    parser.add_argument("--begin", "-b", help="begin stream", type=int, required=True)
    parser.add_argument("--end", "-e", help="end stream", type=int, required=True)
    parser.add_argument("--count", "-n", help="epoch count", type=int, default=12)
    parser.add_argument("--batch-size", "-c", help="batch size", type=int, default=500)
    parser.add_argument("--prefix", "-p", help="dataset prefix", type=str, default="detrac")
    args = parser.parse_args()
    batch_replay(**args.__dict__)
