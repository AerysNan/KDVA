import os
import json
import argparse


def generate_sample_position(sample_count, sample_interval, offset=0):
    sample_win, total = [1 for _ in range(sample_count)], sample_count
    while total < sample_interval:
        for i in range(sample_count):
            sample_win[i] += 1
            total += 1
            if total == sample_interval:
                break
    pos = [offset]
    for i in range(sample_count - 1):
        pos.append(pos[-1] + sample_win[i])
    return pos


def split_dataset(path, dataset, size, train_rate, val_rate, val_size, postfix, **_):
    if train_rate is not None:
        train_sample_count, train_sample_interval = [int(v) for v in train_rate.split('/')]
        sample_pos_train = generate_sample_position(train_sample_count, train_sample_interval, 0)
    if val_rate is not None:
        val_sample_count, val_sample_interval = [int(v) for v in val_rate.split('/')]
        sample_pos_val = generate_sample_position(val_sample_count, val_sample_interval, val_sample_interval // val_sample_count // 2)
    with open('datasets.json') as f:
        datasets = json.load(f)

    epoch_count = datasets[dataset]["size"] // size
    if os.path.exists(f"{path}/{dataset}.gt.json"):
        annotation_all = json.load(open(f"{path}/annotations/{dataset}.gt.json"))
    else:
        annotation_all = None
    annotation_golden = json.load(open(f"{path}/annotations/{dataset}.golden.json"))
    if train_rate is not None:
        annotation_train_list = [
            {"images": [], "annotations": [], "categories": annotation_golden["categories"]}
            for _ in range(epoch_count)
        ]
    if val_rate is not None or val_size is not None:
        annotation_val_list = [
            {"images": [], "annotations": [], "categories": annotation_golden["categories"]}
            for _ in range(epoch_count)
        ]
    annotation_test_gt_list = [
        {"images": [], "annotations": [], "categories": annotation_golden["categories"], "ignored_regions":[]}
        for _ in range(epoch_count)
    ]
    annotation_test_golden_list = [
        {"images": [], "annotations": [], "categories": annotation_golden["categories"], "ignored_regions":[]}
        for _ in range(epoch_count)
    ]
    if annotation_all:
        for image in annotation_all["images"]:
            epoch = image["id"] // size
            offset = image["id"] % size
            annotation_test_gt_list[epoch]["images"].append(image)
        for annotation in annotation_all["annotations"]:
            epoch = annotation["image_id"] // size
            offset = annotation["image_id"] % size
            annotation_test_gt_list[epoch]["annotations"].append(annotation)

    for image in annotation_golden["images"]:
        epoch = image["id"] // size
        offset = image["id"] % size
        annotation_test_golden_list[epoch]["images"].append(image)
        if train_rate is not None and offset % train_sample_interval in sample_pos_train:
            annotation_train_list[epoch]["images"].append(image)
        if val_rate is not None and offset % val_sample_interval in sample_pos_val:
            annotation_val_list[epoch]["images"].append(image)
        if val_size is not None and (val_size > 0 and offset < val_size or val_size < 0 and offset - size >= val_size):
            annotation_val_list[epoch]["images"].append(image)

    for annotation in annotation_golden["annotations"]:
        epoch = annotation["image_id"] // size
        offset = annotation["image_id"] % size
        annotation_test_golden_list[epoch]["annotations"].append(annotation)
        if train_rate is not None and offset % train_sample_interval in sample_pos_train:
            annotation_train_list[epoch]["annotations"].append(annotation)
        if val_rate is not None and offset % val_sample_interval in sample_pos_val:
            annotation_val_list[epoch]["annotations"].append(annotation)
        if val_size is not None and (val_size > 0 and offset < val_size or val_size < 0 and offset - size >= val_size):
            annotation_val_list[epoch]["annotations"].append(annotation)

    for epoch in range(epoch_count):
        if annotation_all and 'ignored_regions' in annotation_all:
            for ignored_region in annotation_all["ignored_regions"]:
                l, r = max(ignored_region['begin'], epoch * size), min(ignored_region['end'], epoch * size + size)
                if l < r:
                    annotation_test_gt_list[epoch]["ignored_regions"].append({
                        "begin": l,
                        "end": r,
                        "region": ignored_region["region"]
                    })
        if train_rate is not None:
            with open(f"{path}/annotations/{dataset}_{postfix}_train_{epoch}.golden.json", "w") as f:
                json.dump(annotation_train_list[epoch], f)
        if val_rate is not None or val_size is not None:
            with open(f"{path}/annotations/{dataset}_{postfix}_val_{epoch}.golden.json", "w") as f:
                json.dump(annotation_val_list[epoch], f)
        with open(f"{path}/annotations/{dataset}_test_{epoch}.golden.json", "w") as f:
            json.dump(annotation_test_golden_list[epoch], f)
        if annotation_all:
            with open(f"{path}/annotations/{dataset}_test_{epoch}.gt.json", "w") as f:
                json.dump(annotation_test_gt_list[epoch], f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate config file")
    parser.add_argument("--path", "-p", help="dataset path", type=str, required=True)
    parser.add_argument("--dataset", "-d", help="dataset name", type=str, required=True)
    parser.add_argument("--size", "-s", help="size of splitted dataset", type=int, default=500)
    parser.add_argument("--train-rate", "-t", help="sampling rate of training dataset", type=str, default=None)
    parser.add_argument("--val-rate", "-r", help="sampling rate of validation dataset", type=str, default=None)
    parser.add_argument("--val-size", "-v", help="sampling size of validation dataset", type=int, default=None)
    parser.add_argument("--postfix", "-o", help="generated postfix", type=str)
    args = parser.parse_args()
    split_dataset(**args.__dict__)
