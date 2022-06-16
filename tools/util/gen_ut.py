import os
import argparse

locations = [
    "Bellevue_116th_NE12th",
    "Bellevue_150th_Eastgate",
    "Bellevue_150th_Newport",
    "Bellevue_150th_SE38th",
    "Bellevue_Bellevue_NE8th"
]

videos = [
    [
        "Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4",
        "Bellevue_116th_NE12th__2017-09-10_23-08-29.mp4",
        "Bellevue_116th_NE12th__2017-09-11_02-08-32.mp4",
        "Bellevue_116th_NE12th__2017-09-11_03-08-30.mp4",
        # "Bellevue_116th_NE12th__2017-09-11_04-08-30.mp4",
        # "Bellevue_116th_NE12th__2017-09-11_06-08-30.mp4"
    ],
    [
        "Bellevue_150th_Eastgate__2017-09-10_18-08-24.mp4",
        "Bellevue_150th_Eastgate__2017-09-10_19-08-25.mp4",
        "Bellevue_150th_Eastgate__2017-09-10_20-08-25.mp4",
        "Bellevue_150th_Eastgate__2017-09-10_21-08-28.mp4",
        # "Bellevue_150th_Eastgate__2017-09-10_22-08-28.mp4",
        # "Bellevue_150th_Eastgate__2017-09-10_23-08-29.mp4"
    ],
    [
        "Bellevue_150th_Newport__2017-09-10_18-08-24.mp4",
        "Bellevue_150th_Newport__2017-09-10_19-08-24.mp4",
        "Bellevue_150th_Newport__2017-09-10_20-08-25.mp4",
        "Bellevue_150th_Newport__2017-09-10_21-08-28.mp4",
        # "Bellevue_150th_Newport__2017-09-10_22-08-28.mp4",
        # "Bellevue_150th_Newport__2017-09-10_23-08-29.mp4"
    ],
    [
        "Bellevue_150th_SE38th__2017-09-10_18-08-24.mp4",
        "Bellevue_150th_SE38th__2017-09-10_19-08-25.mp4",
        "Bellevue_150th_SE38th__2017-09-10_20-08-25.mp4",
        "Bellevue_150th_SE38th__2017-09-10_21-08-38.mp4",
        # "Bellevue_150th_SE38th__2017-09-10_22-08-28.mp4",
        # "Bellevue_150th_SE38th__2017-09-10_23-08-29.mp4"
    ],
    [
        "Bellevue_Bellevue_NE8th__2017-09-10_18-08-23.mp4",
        "Bellevue_Bellevue_NE8th__2017-09-10_19-08-24.mp4",
        "Bellevue_Bellevue_NE8th__2017-09-10_20-08-24.mp4",
        "Bellevue_Bellevue_NE8th__2017-09-10_21-08-28.mp4",
        # "Bellevue_Bellevue_NE8th__2017-09-10_22-08-28.mp4",
        # "Bellevue_Bellevue_NE8th__2017-09-10_23-08-29.mp4"
    ]
]


def generate_ut(input_path, output_path, **_):
    os.makedirs(output_path, exist_ok=True)
    for i, location_videos in videos:
        for j, video in location_videos:
            dataset_name = f'ut_{j * len(videos) + i + 1}'
            os.makedirs(os.path.join(output_path, dataset_name), exist_ok=True)
            os.system(f'ffmpeg -i {os.path.join(input_path, locations[i], video)} -start_number 0 {os.path.join(output_path, dataset_name)}/$filename%06d.jpg')
            if os.path.exists(os.path.join(output_path, dataset_name, '108000.jpg')):
                os.remove(os.path.join(output_path, dataset_name, '108000.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate urban traffic dataset')
    parser.add_argument('--input-path', '-i', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output-path', '-o', type=str, required=True, help='Output dataset path')
    args = parser.parse_args()
    generate_ut(**args.__dict__)
