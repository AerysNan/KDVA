import json
import argparse


def merge_traces(input_file, output_file, **_):
    annotation_id, image_id = 0, 0
    output_annotation = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    datasets = []
    if type(input_file) == list:
        datasets = input_file
    else:
        with open(input_file) as f:
            for line in f:
                datasets.append(line[:-1])
    for dataset in datasets:
        id2id = {}
        with open(dataset) as f:
            dataset_annotation = json.load(f)
        for image in dataset_annotation['images']:
            id2id[image['id']] = image_id
            image['id'] = image_id
            image_id += 1
            output_annotation['images'].append(image)
        for annotation in dataset_annotation['annotations']:
            annotation['image_id'] = id2id[annotation['image_id']]
            annotation['id'] = annotation_id
            annotation_id += 1
            output_annotation['annotations'].append(annotation)
        output_annotation['categories'] = dataset_annotation['categories']
    with open(output_file, 'w') as f:
        json.dump(output_annotation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge traces')
    parser.add_argument('--input-file', '-if', type=str, required=True, help='Input dataset file, each line contains the path to a dataset annotation file')
    parser.add_argument('--output-file', '-of', type=str, required=True, help='Output dataset file')
    args = parser.parse_args()
    merge_traces(**args.__dict__)
