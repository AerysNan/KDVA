from evaluate_from_file import evaluate_from_file
from multiprocessing import Pool
import numpy as np
import argparse
import pickle
import math


def evaluate_validation(path, count, prefix, val, ** _):
    m, m_class = np.zeros((7, 5, 12, count), dtype=np.double), np.zeros((7, 5, 12, count), dtype=np.double)
    p, output = Pool(processes=10), {}

    for retrain in range(7):
        for f in range(5):
            for stream in range(12):
                for epoch in range(count - 1):
                    output[(retrain, f, stream, epoch)] = p.apply_async(evaluate_from_file, (
                        f'{path}/snapshot/result/{prefix}_{stream + 1}_{retrain}_e40v{val}/{epoch + 1:02d}.pkl',
                        f'{path}/data/annotations/{prefix}_{stream + 1}_{val}_val_{epoch}.golden.json', (f + 1, 5),))

    p.close()
    p.join()

    for retrain in range(7):
        for f in range(5):
            for stream in range(12):
                for epoch in range(count - 1):
                    result = output[(retrain, f, stream, epoch)].get()
                    m[retrain, f, stream, epoch + 1] = result["bbox_mAP"]
                    classes_of_interest = ['car']
                    mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
                    m_class[retrain, f, stream, epoch + 1] = sum(mAPs_classwise) / len(mAPs_classwise)

    with open(f'{prefix}_{val}_val.pkl', 'wb') as f:
        pickle.dump({"data": m, "classwise_data": m_class}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate validation result")
    parser.add_argument("--path", "-d", help="data path", type=str, required=True)
    parser.add_argument("--count", "-n", help="epoch count", type=int, default=12)
    parser.add_argument("--prefix", "-p", help="dataset prefix", type=str, default="detrac")
    parser.add_argument("--val", "-v", help="validation postfix", type=str, required=True)
    args = parser.parse_args()
    evaluate_validation(**args.__dict__)
