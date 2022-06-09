import json
import pickle
import argparse


def anno_from_result(root, path, dataset, threshold, **_):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    with open(f'{root}/data/annotations/{dataset}.base.json') as f:
        data = json.load(f)
    data['annotations'] = []
    uid = 0
    for i, frame_result in enumerate(results):
        for j, class_result in enumerate(frame_result):
            for bbox in class_result:
                bbox_list = bbox.tolist()
                if bbox_list[4] < threshold:
                    continue
                annotation = {
                    "id": uid,
                    "image_id": i,
                    "bbox": [bbox_list[0], bbox_list[1], bbox_list[2] - bbox_list[0], bbox_list[3] - bbox_list[1]],
                    "iscrowd": 0,
                    "category_id": j,
                    "area": (bbox_list[2] - bbox_list[0]) * (bbox_list[3] - bbox_list[1])
                }
                data["annotations"].append(annotation)
                uid += 1
    with open(f'{root}/data/annotations/{dataset}.golden.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert result to annotation file')
    parser.add_argument("--root", "-r", help="dataset root", type=str, required=True)
    parser.add_argument('--path', '-p', help='result file path', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='confidence threshold for result filter')
    args = parser.parse_args()
    anno_from_result(**args.__dict__)
