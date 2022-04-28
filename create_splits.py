import argparse
import glob
import os
import random
import numpy as np
import shutil


def split(data_dir, create_test_ds=False):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """

    files = [os.path.join(data_dir, f) for f in os.listdir(
        os.path.abspath(data_dir)) if f.endswith(".tfrecord")]
    if not files:
        print("No file found")
        return

    random.shuffle(files)

    if create_test_ds:
        train, validation, test = list(
            np.split(files, [int(len(files)*0.72), int(len(files)*0.9)]))
    else:
        train, validation = list(np.split(files, [int(len(files)*0.80)]))

    train_folder_path = os.path.join(os.path.dirname(data_dir), "train")
    os.makedirs(train_folder_path, exist_ok=True)
    for t in list(train):
        new_path = os.path.join(train_folder_path, os.path.basename(t))
        if os.path.exists(new_path):
            print("File already exists")
        shutil.move(t, new_path)

    validation_folder_path = os.path.join(os.path.dirname(data_dir), "val")
    os.makedirs(validation_folder_path, exist_ok=True)
    for t in list(validation):
        new_path = os.path.join(validation_folder_path, os.path.basename(t))
        if os.path.exists(new_path):
            print("File already exists")
        shutil.move(t, new_path)

    if create_test_ds:
        test_folder_path = os.path.join(os.path.dirname(data_dir), "test")
        os.makedirs(test_folder_path, exist_ok=True)
        for t in list(test):
            new_path = os.path.join(test_folder_path, os.path.basename(t))
            if os.path.exists(new_path):
                print("File already exists")
            shutil.move(t, new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    parser.add_argument('--add_test_ds', required=False, default=True, nargs='?', const=True,
                        help='Split into 3 dataset train/validation/test instead train/validation')
    args = parser.parse_args()

    split(args.data_dir)
