import json
import os

target_dir_1 = "./aarch64/"
target_dir_2 = "./x86_64/"

target_files_1 = sorted([f for f in os.listdir(target_dir_1) if f.split('.')[-1] == 'json'])
target_files_2 = sorted([f for f in os.listdir(target_dir_2) if f.split('.')[-1] == 'json'])

# target_files_2 = os.listdir(target_dir_2)

print(target_files_1)
print(target_files_2)
