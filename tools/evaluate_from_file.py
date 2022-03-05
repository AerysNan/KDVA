import sys
import json
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


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


def evaluate_from_file(result_path, gt_path, config='configs/custom/ssd_base.py', threshold=0.5):
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
    with open(gt_path) as f:
        gt = json.load(f)
    if "ignored_regions" in gt and len(gt['ignored_regions']) > 0:
        print('Ignored regions detected, start filtering...')
        start = min([image['id'] for image in gt['images']])
        filter_result(result, gt['ignored_regions'], start, threshold)
        print('Filtering finished!')
    return dataset.evaluate(result, metric="bbox", classwise=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDet evaluate from pickle file")
    parser.add_argument(
        "--config", "-c", help="test config file path", default="configs/custom/ssd_base.py"
    )
    parser.add_argument(
        "--result", "-r", help="result file path", type=str, required=True
    )
    parser.add_argument(
        "--gt", "-g", help="ground truth file path", type=str, required=True
    )
    parser.add_argument(
        "--threshold", "-t", help="iou threshold", type=float, default=0.5
    )
    args = parser.parse_args()
    evaluation = evaluate_from_file(args.result, args.gt, args.config, args.threshold)
    print(f'mAP: {evaluation["bbox_mAP"]} classwise: {evaluation["bbox_mAP_car"]}')
