import ast
import json
import pickle
import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Generate config file")
parser.add_argument(
    "--path", "-p", help="path to dataset file", type=str, default="datasets.json"
)
parser.add_argument(
    "--size", "-s", help="size of splitted dataset", type=int, default=600
)
parser.add_argument("--rate", "-r", help="sampling rate of training dataset", type=int)
parser.add_argument("--postfix", "-o", help="generated postfix", type=str)
args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)
if args.postfix:
    postfix = f"_{args.postfix}"
else:
    postfix = ""

epoch_size = args.size
sample_interval = args.rate

for stream_index, prefix in enumerate(dataset):

    epoch_count = dataset[prefix]["size"] // epoch_size

    annotation_all = json.load(open(f"data/annotations/{prefix}.gt.json"))
    annotation_golden = json.load(open(f"data/annotations/{prefix}.golden.json"))

    annotation_train_list = [
        {"images": [], "annotations": [], "categories": annotation_all["categories"]}
        for _ in range(epoch_count)
    ]

    annotation_val_list = [
        {"images": [], "annotations": [], "categories": annotation_all["categories"]}
        for _ in range(epoch_count)
    ]

    annotation_test_list = [
        {"images": [], "annotations": [], "categories": annotation_all["categories"]}
        for _ in range(epoch_count)
    ]

    for image in annotation_all["images"]:
        epoch = image["id"] // epoch_size
        offset = image["id"] % epoch_size
        annotation_test_list[epoch]["images"].append(image)
    for image in annotation_golden["images"]:
        # for image in annotation_all["images"]:
        epoch = image["id"] // epoch_size
        offset = image["id"] % epoch_size
        if offset % sample_interval == 0:
            annotation_train_list[epoch]["images"].append(image)
        if offset % sample_interval == (sample_interval // 2):
            annotation_val_list[epoch]["images"].append(image)

    for annotation in annotation_all["annotations"]:
        epoch = annotation["image_id"] // epoch_size
        offset = annotation["image_id"] % epoch_size
        annotation_test_list[epoch]["annotations"].append(annotation)
    for annotation in annotation_golden["annotations"]:
        # for annotation in annotation_all["annotations"]:
        epoch = annotation["image_id"] // epoch_size
        offset = annotation["image_id"] % epoch_size
        if offset % sample_interval == 0:
            annotation_train_list[epoch]["annotations"].append(annotation)
        elif offset % sample_interval == (sample_interval // 2):
            annotation_val_list[epoch]["annotations"].append(annotation)

    for epoch in range(epoch_count):
        with open(f"data/annotations/{prefix}{postfix}_train_{epoch}.golden.json", "w") as f:
            json.dump(annotation_train_list[epoch], f)
        with open(f"data/annotations/{prefix}{postfix}_test_{epoch}.gt.json", "w") as f:
            json.dump(annotation_test_list[epoch], f)
        with open(f"data/annotations/{prefix}{postfix}_val_{epoch}.golden.json", "w") as f:
            json.dump(annotation_val_list[epoch], f)
