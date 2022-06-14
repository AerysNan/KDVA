from evaluate_from_file import evaluate_from_file
from multiprocessing import Pool
import numpy as np
import argparse
import pickle
import math
import ast

streams = {
    1: '1m12',
    2: '2m12',
    3: '3m34',
    4: '4m34',
}


def evaluate_validation(path, epoch, prefix, retrain_l, retrain_r, infer_l, infer_r, stream_l, stream_r, postfix, val, n_process, ** _):
    if val:
        m = np.zeros((retrain_r - retrain_l + 1, infer_r - infer_l + 1, stream_r - stream_l + 1, epoch), dtype=np.double)
        m_class = np.zeros((retrain_r - retrain_l + 1, infer_r - infer_l + 1, stream_r - stream_l + 1, epoch), dtype=np.double)
        p, output = Pool(processes=n_process), {}
        for retrain in range(retrain_l, retrain_r + 1):
            for infer in range(infer_l, infer_r + 1):
                for stream in range(stream_l, stream_r + 1):
                    _stream = streams[stream] if postfix == 'agg' else stream
                    _postfix = f'{stream}_base' if retrain == 0 else f'{_stream}_{retrain}_{postfix}'
                    for e in range(epoch - 1):
                        output[(retrain, infer, stream, e)] = p.apply_async(evaluate_from_file, (
                            f'{path}/snapshot/result/{prefix}_{_postfix}v/{e + 1:02d}.pkl',
                            f'{path}/data/annotations/{prefix}_{stream}_val_{e}.golden.json', (infer + 1, 5),))
        p.close()
        p.join()
        for retrain in range(retrain_l, retrain_r + 1):
            for infer in range(infer_l, infer_r + 1):
                for stream in range(stream_l, stream_r + 1):
                    for e in range(epoch - 1):
                        result = output[(retrain, infer, stream, e)].get()
                        if "bbox_mAP" in result:
                            m[retrain - retrain_l, infer - infer_l, stream - stream_l, e + 1] = result["bbox_mAP"]
                        else:
                            m[retrain - retrain_l, infer - infer_l, stream - stream_l, e + 1] = -1
                        classes_of_interest = ['car']
                        if "classwise" in result:
                            mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
                            m_class[retrain - retrain_l, infer - infer_l, stream - stream_l, e + 1] = sum(mAPs_classwise) / len(mAPs_classwise) if len(mAPs_classwise) > 0 else -1
                        else:
                            m_class[retrain - retrain_l, infer - infer_l, stream - stream_l, e + 1] = -1
        with open(f'configs/cache/{prefix}_{postfix}_val.pkl', 'wb') as f:
            pickle.dump({"data": m, "classwise_data": m_class}, f)
    else:
        m = np.zeros((retrain_r - retrain_l + 1, infer_r - infer_l + 1, stream_r - stream_l + 1, epoch + 1), dtype=np.double)
        m_class = np.zeros((retrain_r - retrain_l + 1, infer_r - infer_l + 1, stream_r - stream_l + 1, epoch + 1), dtype=np.double)
        # p, output = Pool(processes=n_process), {}
        output = {}
        for retrain in range(retrain_l, retrain_r + 1):
            for infer in range(infer_l, infer_r + 1):
                for stream in range(stream_l, stream_r + 1):
                    _stream = streams[stream] if postfix == 'agg' else stream
                    _postfix = f'{stream}_base' if retrain == 0 else f'{_stream}_{retrain}_{postfix}'
                    for e in range(epoch):
                        # output[(retrain, infer, stream, e)] = p.apply_async(evaluate_from_file, (
                        #     f'{path}/snapshot/result/{prefix}_{_postfix}/{e:02d}.pkl',
                        #     f'{path}/data/annotations/{prefix}_{stream}_test_{e}.golden.json', (infer + 1, 5),))
                        output[(retrain, infer, stream, e)] = evaluate_from_file(
                            f'{path}/snapshot/result/{prefix}_{_postfix}/{e:02d}.pkl',
                            f'{path}/data/annotations/{prefix}_{stream}_test_{e}.golden.json', (infer + 1, 5))
                    # output[(retrain, infer, stream, epoch)] = p.apply_async(evaluate_from_file, (
                    #     f'{path}/snapshot/result/{prefix}_{_postfix}',
                    #     f'{path}/data/annotations/{prefix}_{stream}.golden.json', (infer + 1, 5), True, ))
                    output[(retrain, infer, stream, epoch)] = evaluate_from_file(
                        f'{path}/snapshot/result/{prefix}_{_postfix}',
                        f'{path}/data/annotations/{prefix}_{stream}.golden.json', (infer + 1, 5), True)
        # p.close()
        # p.join()
        for retrain in range(retrain_l, retrain_r + 1):
            for infer in range(infer_l, infer_r + 1):
                for stream in range(stream_l, stream_r + 1):
                    for e in range(epoch + 1):
                        # result = output[(retrain, infer, stream, e)].get(3600)
                        result = output[(retrain, infer, stream, e)]
                        m[retrain - retrain_l, infer - infer_l, stream - stream_l, e] = result["bbox_mAP"]
                        classes_of_interest = ['car']
                        mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
                        m_class[retrain - retrain_l, infer - infer_l, stream - stream_l, e] = sum(mAPs_classwise) / len(mAPs_classwise)
        with open(f'configs/cache/{prefix}_{postfix}_{stream_l}_{stream_r}.pkl', 'wb') as f:
            pickle.dump({"data": m, "classwise_data": m_class}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate validation result")
    parser.add_argument("--path", "-d", help="data path", type=str, required=True)
    parser.add_argument("--epoch", "-n", help="epoch count", type=int, default=12)
    parser.add_argument("--prefix", "-p", help="dataset prefix", type=str, default="detrac")
    parser.add_argument('--stream-l', '-sl', type=int, default=1)
    parser.add_argument('--stream-r', '-sr', type=int, default=12)
    parser.add_argument('--retrain-l', '-rl', type=int, default=0)
    parser.add_argument('--retrain-r', '-rr', type=int, default=5)
    parser.add_argument('--infer-l', '-il', type=int, default=0)
    parser.add_argument('--infer-r', '-ir', type=int, default=4)
    parser.add_argument('--n-process', '-np', type=int, default=4)
    parser.add_argument("--postfix", "-o", type=str, default="e40")
    parser.add_argument('--val', '-v', type=ast.literal_eval, default=True)
    args = parser.parse_args()
    evaluate_validation(**args.__dict__)
