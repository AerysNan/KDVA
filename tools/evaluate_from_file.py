import os
import sys
import ast
import json
import math
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset
from split_dataset import generate_sample_position


def intersect(rec1, rec2):
    return not (rec1[2] <= rec2[0] or
                rec1[3] <= rec2[1] or
                rec1[0] >= rec2[2] or
                rec1[1] >= rec2[3])


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    if boxAArea == 0:
        return 1
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / float(boxAArea)
    return iou


def filter_result(result, ignored_regions, start, threshold=0.5):
    for region in ignored_regions:
        for i in range(region['begin'] - start, region['end'] - start):
            for j, class_result in enumerate(result[i]):
                indices = []
                for bbox in class_result:
                    indices.append(iou(bbox, region['region']) < threshold)
                result[i][j] = class_result[indices]


def evaluate_from_file(result_path, gt_path, downsample=None, merge=False, config='configs/custom/ssd_base.py', threshold=0.5, **_):
    cfg = Config.fromfile(config)
    cfg.data.test.ann_file = gt_path
    cfg.data.test.img_prefix = ""
    dataset = build_dataset(cfg.data.test)
    if type(result_path) == str:
        if not merge:
            with open(result_path, "rb") as f:
                result = pickle.load(f)
        else:
            result = []
            files = os.listdir(result_path)
            files.sort()
            for file in files:
                with open(f'{result_path}/{file}', 'rb') as f:
                    result.extend(pickle.load(f))
    elif type(result_path) == list:
        result = result_path
    else:
        print('Unsupported result format!')
        sys.exit(1)
    if type(result) == list and type(result[0]) != list:
        result = [result]
    with open(gt_path) as f:
        gt = json.load(f)
    if "ignored_regions" in gt and len(gt['ignored_regions']) > 0:
        print('Ignored regions detected, start filtering...')
        start = min([image['id'] for image in gt['images']])
        filter_result(result, gt['ignored_regions'], start, threshold)
        print('Filtering finished!')
    if downsample:
        positions = generate_sample_position(downsample[0], downsample[1])
        for start in range(0, len(result), downsample[1]):
            for j in range(len(positions) - 1):
                for k in range(positions[j] + 1, positions[j + 1]):
                    result[start + k] = result[start + positions[j]]
            for k in range(positions[-1] + 1, downsample[1]):
                result[start + k] = result[start + positions[-1]]
    return dataset.evaluate(result, metric="bbox", classwise=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDet evaluate from pickle file")
    parser.add_argument("--config", "-c", help="test config file path", default="configs/custom/ssd_base.py")
    parser.add_argument("--result-path", "-r", help="result file path", type=str, required=True)
    parser.add_argument("--gt-path", "-g", help="ground truth file path", type=str, required=True)
    parser.add_argument("--threshold", "-t", help="iou threshold", type=float, default=0.5)
    parser.add_argument("--downsample", "-d", help="downsample rate", type=str, default=None)
    parser.add_argument("--merge", "-m", help="merge results", type=ast.literal_eval, default=False)
    args = parser.parse_args()
    if args.downsample is not None:
        downsample = args.downsample.split('/')
        args.downsample = (int(downsample[0]), int(downsample[1]))
    evaluation = evaluate_from_file(**args.__dict__)
    # classes_of_interest = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    classes_of_interest = ['car']
    mAPs_classwise = [evaluation["classwise"][c] for c in classes_of_interest if not math.isnan(evaluation["classwise"][c])]
    print(f'mAP: {evaluation["bbox_mAP"]} classwise: {sum(mAPs_classwise) / len(mAPs_classwise) if len(mAPs_classwise) > 0 else -1:.3f}')
