"""
This file generate a csv file containing filename and label of each data image. 
"""

import argparse
import os
import csv
import sys

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Please install sklearn:\n\n  pip install scikit-learn\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentDefaultsHelpFormatter(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data", help="image data directory")
    parser.add_argument("output", help="output directory")
    args = parser.parse_args()

    data = args.data
    output = args.output

    no_algae_path = os.path.join(data, "no-algae")
    algae_path = os.path.join(data, "algae")
    output_csv = os.path.join(output, "algae_classification.csv")
    filename_arr = []

    for image_name in os.listdir(no_algae_path):
        filename_arr.append([image_name, 'no algae'])

    for image_name in os.listdir(algae_path):
        filename_arr.append([image_name, 'got algae'])

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'status'])
        for f in filename_arr:
            writer.writerow(f)

if __name__ == "__main__":
    main()
