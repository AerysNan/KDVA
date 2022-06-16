import os
import json
import argparse

TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'


def generate_sample_position(n_samples, n_frames, offset=0):
    sample_win, total = [1 for _ in range(n_samples)], n_samples
    while total < n_frames:
        for i in range(n_samples):
            sample_win[i] += 1
            total += 1
            if total == n_frames:
                break
    pos = [offset]
    for i in range(n_samples - 1):
        pos.append(pos[-1] + sample_win[i])
    return pos


def split_dataset(input_file, output_dir, output_name, size, train_rate=None, val_rate=None, val_size=None, **_):
    if train_rate is not None:
        train_sample_count, train_sample_interval = [int(v) for v in train_rate.split('/')]
        sample_pos_train = generate_sample_position(train_sample_count, train_sample_interval, 0)
    if val_rate is not None:
        val_sample_count, val_sample_interval = [int(v) for v in val_rate.split('/')]
        sample_pos_val = generate_sample_position(val_sample_count, val_sample_interval, val_sample_interval // val_sample_count // 2)

    annotation_golden = json.load(open(input_file))
    epoch_count = len(annotation_golden['images']) // size

    if train_rate is not None:
        annotation_train_list = [{"images": [], "annotations": [], "categories": annotation_golden["categories"]} for _ in range(epoch_count)]
    if val_rate is not None or val_size is not None:
        annotation_val_list = [{"images": [], "annotations": [], "categories": annotation_golden["categories"]} for _ in range(epoch_count)]
    annotation_test_list = [{"images": [], "annotations": [], "categories": annotation_golden["categories"], "ignored_regions":[]} for _ in range(epoch_count)]

    for image in annotation_golden["images"]:
        epoch = image["id"] // size
        offset = image["id"] % size
        annotation_test_list[epoch]["images"].append(image)
        if train_rate is not None and offset % train_sample_interval in sample_pos_train:
            annotation_train_list[epoch]["images"].append(image)
        if val_rate is not None and offset % val_sample_interval in sample_pos_val:
            annotation_val_list[epoch]["images"].append(image)
        if val_size is not None and (val_size > 0 and offset < val_size or val_size < 0 and offset - size >= val_size):
            annotation_val_list[epoch]["images"].append(image)

    for annotation in annotation_golden["annotations"]:
        epoch = annotation["image_id"] // size
        offset = annotation["image_id"] % size
        annotation_test_list[epoch]["annotations"].append(annotation)
        if train_rate is not None and offset % train_sample_interval in sample_pos_train:
            annotation_train_list[epoch]["annotations"].append(annotation)
        if val_rate is not None and offset % val_sample_interval in sample_pos_val:
            annotation_val_list[epoch]["annotations"].append(annotation)
        if val_size is not None and (val_size > 0 and offset < val_size or val_size < 0 and offset - size >= val_size):
            annotation_val_list[epoch]["annotations"].append(annotation)

    os.makedirs(os.path.join(output_dir, TRAIN_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_dir, TEST_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_dir, VAL_DIR), exist_ok=True)
    for epoch in range(epoch_count):
        if train_rate is not None:
            with open(os.path.join(output_dir, TRAIN_DIR, f'{output_name}.{epoch}'), "w") as f:
                json.dump(annotation_train_list[epoch], f)
        if val_rate is not None or val_size is not None:
            with open(os.path.join(output_dir, VAL_DIR, f'{output_name}.{epoch}'), "w") as f:
                json.dump(annotation_val_list[epoch], f)
        with open(os.path.join(output_dir, TEST_DIR, f'{output_name}.{epoch}'), "w") as f:
            json.dump(annotation_test_list[epoch], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate config file")
    parser.add_argument("--input-file", "-i", help="input file path", type=str, required=True)
    parser.add_argument("--output-name", "-on", help="output name", type=str, required=True)
    parser.add_argument("--output-dir", "-od", help="output directory", type=str, required=True)
    parser.add_argument("--size", "-s", help="size of splitted dataset", type=int, default=500)
    parser.add_argument("--train-rate", "-t", help="sampling rate of training dataset", type=str, default=None)
    parser.add_argument("--val-rate", "-r", help="sampling rate of validation dataset", type=str, default=None)
    parser.add_argument("--val-size", "-v", help="sampling size of validation dataset", type=int, default=None)
    args = parser.parse_args()
    split_dataset(**args.__dict__)
