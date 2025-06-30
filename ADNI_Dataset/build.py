import os
import time
import datetime

import numpy as np
import pandas as pd

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
#import tensorflow_hub as hub
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import glob
from misc_mason import *

import shutil

folder_path = '/data/datasets/AD/raw_data/'
file_extension = 'nii'
details = list_files(folder_path, file_extension)
print(details['non-nii']['file_paths'])
details['all_file_paths']

def organize_images(source_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Exclude the .ipynb_checkpoints directory
        if '.ipynb_checkpoints' in dirs:
            dirs.remove('.ipynb_checkpoints')

        # Iterate through the files
        for file in files:
            file_path = os.path.join(root, file)

            if 'AD' in file:
                class_label = 'AD'
            elif 'MCI' in file:
                class_label = 'MCI'
            elif 'CN' in file:
                class_label = 'CN'

            # Create the class directory in the output directory if it doesn't exist
            class_dir = os.path.join(output_dir, class_label)
            os.makedirs(class_dir, exist_ok=True)

            # Move the file to the class directory
            dst_path = os.path.join(class_dir, file)
            shutil.move(file_path, dst_path)

    print("Image organization completed.")


# Convert NII to PNG and create the ADNI_IMG_32.5%_x_unprocessed folder
folder_path = '/data/datasets/AD/raw_data/'
details = list_files(folder_path, 'nii')
all_files = details['all_file_paths']

enum_len = len(list(enumerate(all_files)))

for i, file in enumerate(all_files):
    i += 1
    if '.nii' in file:
        print(file)
        try:
            convert_nii_to_png(file, 0.325, 'x')
            print(f"Done with ({i}/{enum_len})")
        except KeyError:
            # Get the filename without extension
            filename, _ = os.path.splitext(os.path.basename(file))

            # Split the filename by underscores and get the last part
            filename.split('_')[-1]

            print(filename, "NOT found in .csv, skipping!")

print(("-" * 25) + "\nDone with creating folder")
