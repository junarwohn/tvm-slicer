import cv2
import os
import shutil
import random
import numpy as np

dataset_types = ['train', 'validation', 'test']
dataset_idx = 2

dataset_type = dataset_types[dataset_idx]


raw_data_path = "./data/us-bmod/{}/raw/0/".format(dataset_type)
mask_data_path = "./data/us-bmod/{}/mask/0/".format(dataset_type)


data_file_lists = [fpath for fpath in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, fpath))]

print(sorted(data_file_lists))

for file_name in data_file_lists:
    raw = cv2.resize(cv2.imread(os.path.join(raw_data_path, file_name)), (512, 512))
    ret, mask = cv2.threshold(cv2.resize(cv2.imread(os.path.join(mask_data_path, file_name)), (512, 512)), 20, 255, cv2.THRESH_BINARY)
    result = np.hstack((raw, mask))
    print(file_name)
    cv2.imshow("show", result)
    cv2.waitKey(0)
