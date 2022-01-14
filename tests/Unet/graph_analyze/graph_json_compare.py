import json
import os

from numpy.lib.arraysetops import isin

target_dir_1 = "./aarch64/"
target_dir_2 = "./x86_64/"

target_files_1 = sorted([f for f in os.listdir(target_dir_1) if f.split('.')[-1] == 'json'])
target_files_2 = sorted([f for f in os.listdir(target_dir_2) if f.split('.')[-1] == 'json'])

# target_files_2 = os.listdir(target_dir_2)

print(target_files_1)
print(target_files_2)

def compare(data_1, data_2):
    if isinstance(data_1, dict) and isinstance(data_2, dict):
        data_1.sort()
        data_2.sort()
        result = True
        for key_1, key_2 in zip(data_1, data_2):
            if key_1 != key_2:
                raise Exception("Key is not same", key_1, key_2)
            else:
                result &= compare(data_1[key_1], data_2[key_2])
    else:
        result = (data_1 == data_2)
    return result

for i in range(len(target_files_1)):
    target_json_file_path_1 = target_dir_1 + target_files_1[i]
    target_json_file_path_2 = target_dir_2 + target_files_2[i]

    with open(target_json_file_path_1, "r") as json_graph_1:
        json_graph_1 = json.load(json_graph_1)
    with open(target_json_file_path_2, "r") as json_graph_2:
        json_graph_2 = json.load(json_graph_2)
        
    print(target_files_1[i], target_files_2[i], compare(json_graph_1, json_graph_2))
    