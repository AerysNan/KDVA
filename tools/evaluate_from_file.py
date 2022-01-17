import sys
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def evaluate_from_file(result_path, gt_path, config='configs/custom/ssd.py'):
    cfg = Config.fromfile(config)
    cfg.data.test.ann_file = gt_path
    cfg.data.test.img_prefix = ""
    dataset = build_dataset(cfg.data.test)
    if type(result_path) == str:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
    elif type(result_path) == list:
        result = result_path
    else:
        print('Unsupported result format!')
        sys.exit(1)
    if type(result) == list and type(result[0]) != list:
        result = [result]
    return dataset.evaluate(result, metric="bbox")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDet evaluate from pickle file")
    parser.add_argument(
        "--config", "-c", help="test config file path", default="configs/custom/ssd.py"
    )
    parser.add_argument(
        "--result", "-r", help="result file path", type=str, required=True
    )
    parser.add_argument(
        "--gt", "-g", help="ground truth file path", type=str, required=True
    )
    args = parser.parse_args()
    print(evaluate_from_file(args.result, args.gt, args.config)["bbox_mAP"])
