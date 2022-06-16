import json
import argparse


def generate_detrac(input_file, output_file, **_):
    with open(input_file) as f:
        gt = json.load(f)
    classes_of_interest = {3, 6, 8}
    gt['categories'] = [{'supercategory': 'none', 'id': 1, 'name': 'vehicle'}]
    annotations, image_ids = [], set()
    for annotation in gt['annotations']:
        if annotation['category_id'] in classes_of_interest:
            annotation['category_id'] = 1
            annotations.append(annotation)
            image_ids.add(annotation['image_id'])
    images = []
    for image in gt['images']:
        if image['id'] in image_ids:
            images.append(image)
    gt['images'] = images
    gt['annotations'] = annotations
    with open(output_file, 'w') as f:
        json.dump(gt, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate DETRAC dataset')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output-file', '-o', type=str, required=True, help='Output dataset path')
    args = parser.parse_args()
    generate_detrac(**args.__dict__)
