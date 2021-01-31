"""
Extract frames from a video file.
"""

import cv2
import os.path as osp
import os
import argparse


def extract_frames(input_dir, output_dir):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    filename_ext = osp.basename(input_dir)
    filename = osp.splitext(filename_ext)[0]
    count = 0
    vidcap = cv2.VideoCapture(input_dir)
    while True:
        success, image = vidcap.read()
        if not success:
            break
        output_filename = '%s-frame-%d.jpg' % (filename, count)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        print('Output file: ', output_filename)
        cv2.imwrite(os.path.join(output_dir, output_filename), image)  # save frame as JPEG file
        count = count + 1


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input_dir", help="input directory", required=True)
    a.add_argument("--output_dir", help="output directory", required=True)
    args = a.parse_args()
    extract_frames(args.input_dir, args.output_dir)
