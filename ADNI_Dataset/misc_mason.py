# Imports

import os
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


#Constants

RED = "\033[0;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


# Available Memory

def memory():
    import psutil
    print(round(psutil.virtual_memory().available / (1024**3),2), 'of', round(psutil.virtual_memory().total / (1024**3),2), 'GB available')


# Zip a folder

def zip_folder(original_folder_name, zip_file_name):
    import shutil
    shutil.make_archive(zip_file_name, 'zip', original_folder_name)
    print('-'*80)
    print('Done with zipping the file')


# Get folder details
    
def folder_details(file_path, units='MB'):
    folder_size = f'folder_size_{units}'
    details = {folder_size: 0,
               'num_files': 0
              }
    units_numbers = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            details[folder_size] += os.path.getsize(os.path.join(root, file))
            details['num_files'] += 1

    details[folder_size] /= (1024 ** units_numbers[units])
    
    return details


# List files

def list_files(folder_path, file_type):
    non_file_type = 'non-' + file_type
    details = {
        'all_file_paths' : [],
        file_type : {
            'num_files' : 0,
            'file_size_MB' : 0
        },
        non_file_type : {
            'num_files' : 0,
            'file_size_MB' : 0,
            'file_paths' : []
        }
    }
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / (10**6)
            if file_type in file_path:
                details[file_type]['num_files'] += 1
                details[file_type]['file_size_MB'] += file_size
                details['all_file_paths'].append(file_path)
            else:
#                 print(RED,file_path,RESET,sep='')
                details[non_file_type]['num_files'] += 1
                details[non_file_type]['file_size_MB'] += file_size
                details[non_file_type]['file_paths'].append(file_path)
    return details


# Converting nii files to PNG
def convert_nii_to_png(nii_file_path, percentile, dim_label):
    print(nii_file_path)
    if os.path.isdir(nii_file_path):
        return
    
    image_data_id = nii_file_path.split('/')[10]
    subject = nii_file_path.split('/')[7]
    print("image_data_id",image_data_id)
    
    df = pd.read_csv('/data/datasets/AD/raw_data/ADNI1_Complete_3Yr_1.5T/ADNI1_Complete_3Yr_1.5T_7_21_2023.csv')
    ad_cn_mci = {i for i in df.loc[df['Image Data ID'] == image_data_id].Group}.pop()
    
    output_folder = "ADNI_IMG_" + str(round(percentile * 100, 1)) + f"%_{dim_label}"
    nifti_image = nib.load(nii_file_path)
    image_data = nifti_image.get_fdata()
    image_data = normalize_image(image_data)  # Normalize image data
    
    os.makedirs(output_folder, exist_ok=True)
    num_slices = image_data.shape[{'x':0,'y':1,'z':2}.get(dim_label.lower(),0)]

    index = int(num_slices * percentile)

    if dim_label == 'x':
        slice_data = np.squeeze(image_data[index, :, :])
    elif dim_label == 'y':
        slice_data = np.squeeze(image_data[:, index, :])
    else:
        slice_data = np.squeeze(image_data[:, :, index])

    slice_data = scale_image(slice_data)
    slice_data = (slice_data * 255).astype(np.uint8)

    slice_image = Image.fromarray(slice_data)
    slice_image = slice_image.convert('RGB')
    
    output_path = os.path.join(output_folder, f"{subject}__{image_data_id}__{ad_cn_mci}__.png")
    slice_image.save(output_path)
    
    change_size((224,224), output_folder)
    convert_grayscale_rgb(output_folder)


def normalize_image(image_data):
    min_value = np.min(image_data)
    max_value = np.max(image_data)
    normalized_data = (image_data - min_value) / (max_value - min_value)
    return normalized_data

def scale_image(slice_data):
    percentile_1 = np.percentile(slice_data, 1)
    percentile_99 = np.percentile(slice_data, 99)
    scaled_data = (slice_data - percentile_1) / (percentile_99 - percentile_1)
    scaled_data = np.clip(scaled_data, 0, 1)
    return scaled_data

def change_size(IMAGE_SHAPE, output_folder):
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Open the image and resize it
            image = Image.open(file_path).resize(IMAGE_SHAPE)

            # Save the resized image
            image.save(file_path)

    print("Image resizing completed.")

def convert_grayscale_rgb(output_folder):
    converted_images = []
    for file in os.listdir(output_folder):
        # make full file path
        file_path = os.path.join(output_folder, file)
        
        # Open the grayscale image
        grayscale_image = Image.open(file_path)

        # Convert the grayscale image to RGB
        rgb_image = grayscale_image.convert('RGB')

        # Save the RGB image
        rgb_image.save(file_path)

# if __name__ == '__main__':
#     folder_path = '/data/datasets/AD'
#     details = list_files(folder_path, 'nii')
#     all_files = details['all_file_paths']

#     for file in all_files:
#         if '.nii' in file:
#             convert_nii_to_png(file, 0.325, 'x')
            
#     print(("-"*25)+"\nDone with creating folder")