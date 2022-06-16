import os
import argparse
from shutil import copyfile

videos = [
    [
        'MVI_39761',
        'MVI_39771',
        'MVI_39801',
        'MVI_39811',
        'MVI_39821',
        'MVI_39861',
        'MVI_40141',
        'MVI_40152',
        'MVI_63544'
    ],
    [
        'MVI_20011',
        'MVI_20012',
        'MVI_20032',
        'MVI_20033',
        'MVI_20034',
        'MVI_20035',
        'MVI_20051',
        'MVI_20052',
        'MVI_20061',
        'MVI_20062',
        'MVI_20063',
        'MVI_20064',
        'MVI_20065',
        'MVI_39781',
        'MVI_39851',
        'MVI_40131',
        'MVI_40161',
        'MVI_40162',
        'MVI_40171',
        'MVI_40172',
        'MVI_40181',
        'MVI_63521',
        'MVI_63525'
    ],
    [
        'MVI_40191',
        'MVI_40192',
        'MVI_40201',
        'MVI_40204',
        'MVI_40211',
        'MVI_40212',
        'MVI_40213',
        'MVI_40241',
        'MVI_40243',
        'MVI_40244',
        'MVI_41063',
        'MVI_41073',
        'MVI_63552',
        'MVI_63553',
        'MVI_63554',
        'MVI_63561',
        'MVI_63562',
        'MVI_63563'
    ],
    [
        'MVI_40701',
        'MVI_40851',
        'MVI_40852',
        'MVI_40853',
        'MVI_40854',
        'MVI_40855',
        'MVI_40891',
        'MVI_40892',
        'MVI_40901',
        'MVI_40902',
        'MVI_40903',
        'MVI_40904',
        'MVI_40905'
    ],
    [
        'MVI_40742',
        'MVI_40743',
        'MVI_40863',
        'MVI_40864'
    ],
    [
        'MVI_40771',
        'MVI_40772',
        'MVI_40773',
        'MVI_40774',
        'MVI_40775',
        'MVI_40792',
        'MVI_40793'
    ]
]


def generate_detrac(input_path, output_path, **_):
    SIZE = 6000
    for i in range(6):
        dataset_name = f'detrac_{i + 1}'
        os.makedirs(os.path.join(output_path, dataset_name), exist_ok=True)
        image_id = 0
        for video in videos[i]:
            if image_id >= SIZE:
                break
            files = os.listdir(os.path.join(input_path, video))
            files.sort()
            for file in files:
                copyfile(os.path.join(input_path, video, file), os.path.join(output_path, dataset_name, f'{image_id:06d}.jpg'))
                image_id += 1
                if image_id >= SIZE:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate DETRAC dataset')
    parser.add_argument('--input-path', '-i', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output-path', '-o', type=str, required=True, help='Output dataset path')
    args = parser.parse_args()
    generate_detrac(**args.__dict__)
