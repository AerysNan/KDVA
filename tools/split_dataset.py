import os
import json
import argparse

parser = argparse.ArgumentParser(description="Generate config file")
parser.add_argument(
    "--path", "-p", help="path to dataset file", type=str, default="datasets.json"
)
parser.add_argument(
    "--size", "-s", help="size of splitted dataset", type=int, default=500
)
parser.add_argument("--rate", "-r", help="sampling rate of training dataset", type=str)
parser.add_argument("--postfix", "-o", help="generated postfix", type=str)

args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)
if args.postfix:
    postfix = f"_{args.postfix}"
else:
    postfix = ""

epoch_size = args.size


def generate_sample_position(sample_count, sample_interval):
    sample_win, total = [1 for _ in range(sample_count)], sample_count
    while total < sample_interval:
        for i in range(sample_count):
            sample_win[i] += 1
            total += 1
            if total == sample_interval:
                break
    pos_test, pos_val = [0], [(sample_win[0] + 1) // 2]
    for i in range(sample_count - 1):
        pos_test.append(pos_test[-1] + sample_win[i])
        pos_val.append(pos_test[-1] + (sample_win[i + 1] + 1) // 2)
    return pos_test, pos_val


sample_count, sample_interval = [int(v) for v in args.rate.split('/')]
sample_pos_test, sample_pos_val = generate_sample_position(sample_count, sample_interval)

for stream_index, prefix in enumerate(dataset):

    epoch_count = dataset[prefix]["size"] // epoch_size
    if os.path.exists(f"data/annotations/{prefix}.gt.json"):
        annotation_all = json.load(open(f"data/annotations/{prefix}.gt.json"))
    else:
        annotation_all = None
    annotation_golden = json.load(open(f"data/annotations/{prefix}.golden.json"))

    annotation_train_list = [
        {"images": [], "annotations": [], "categories": annotation_golden["categories"]}
        for _ in range(epoch_count)
    ]

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
            epoch = image["id"] // epoch_size
            offset = image["id"] % epoch_size
            annotation_test_gt_list[epoch]["images"].append(image)

    for image in annotation_golden["images"]:
        epoch = image["id"] // epoch_size
        offset = image["id"] % epoch_size
        annotation_test_golden_list[epoch]["images"].append(image)

    for image in annotation_golden["images"]:
        epoch = image["id"] // epoch_size
        offset = image["id"] % epoch_size
        if offset % sample_interval in sample_pos_test:
            annotation_train_list[epoch]["images"].append(image)
        if offset % sample_interval in sample_pos_val:
            annotation_val_list[epoch]["images"].append(image)
    if annotation_all:
        for annotation in annotation_all["annotations"]:
            epoch = annotation["image_id"] // epoch_size
            offset = annotation["image_id"] % epoch_size
            annotation_test_gt_list[epoch]["annotations"].append(annotation)

    for annotation in annotation_golden["annotations"]:
        epoch = annotation["image_id"] // epoch_size
        offset = annotation["image_id"] % epoch_size
        annotation_test_golden_list[epoch]["annotations"].append(annotation)

    for annotation in annotation_golden["annotations"]:
        epoch = annotation["image_id"] // epoch_size
        offset = annotation["image_id"] % epoch_size
        if offset % sample_interval in sample_pos_test:
            annotation_train_list[epoch]["annotations"].append(annotation)
        elif offset % sample_interval in sample_pos_val:
            annotation_val_list[epoch]["annotations"].append(annotation)
    for epoch in range(epoch_count):
        if annotation_all and 'ignored_regions' in annotation_all:
            for ignored_region in annotation_all["ignored_regions"]:
                l, r = max(ignored_region['begin'], epoch * epoch_size), min(ignored_region['end'], epoch * epoch_size + epoch_size)
                if l < r:
                    annotation_test_gt_list[epoch]["ignored_regions"].append({
                        "begin": l,
                        "end": r,
                        "region": ignored_region["region"]
                    })
        with open(f"data/annotations/{prefix}{postfix}_train_{epoch}.golden.json", "w") as f:
            json.dump(annotation_train_list[epoch], f)
        if annotation_all:
            with open(f"data/annotations/{prefix}_test_{epoch}.gt.json", "w") as f:
                json.dump(annotation_test_gt_list[epoch], f)
        with open(f"data/annotations/{prefix}_test_{epoch}.golden.json", "w") as f:
            json.dump(annotation_test_golden_list[epoch], f)
        with open(f"data/annotations/{prefix}{postfix}_val_{epoch}.golden.json", "w") as f:
            json.dump(annotation_val_list[epoch], f)
