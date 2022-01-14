import json
import os

from numpy.lib.arraysetops import isin

target_dir_1 = "./x86_64/"
target_dir_2 = "./aarch64/"

target_files_1 = sorted([f for f in os.listdir(target_dir_1) if f.split('.')[-1] == 'json'])
target_files_2 = sorted([f for f in os.listdir(target_dir_2) if f.split('.')[-1] == 'json'])

def compare(data_1, data_2):
    if isinstance(data_1, dict) and isinstance(data_2, dict):
        data_1_keys = sorted(data_1)
        data_2_keys = sorted(data_2)
        result = True
        for key_1, key_2 in zip(data_1_keys, data_2_keys):
            if key_1 != key_2:
                raise Exception("Key is not same", key_1, key_2)
            else:
                # print(key_1, key_2)
                if key_1 == 'storage_id' or key_1 == "device_index":
                    continue
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
        
    print(target_dir_1+target_files_1[i], target_dir_2+target_files_2[i], compare(json_graph_1, json_graph_2))

for i in range(len(target_files_1)):
    target_json_file_path_1 = target_dir_1 + target_files_1[i]
    target_json_file_path_2 = target_dir_2 + target_files_2[(i + 4) % 8]

    with open(target_json_file_path_1, "r") as json_graph_1:
        json_graph_1 = json.load(json_graph_1)
    with open(target_json_file_path_2, "r") as json_graph_2:
        json_graph_2 = json.load(json_graph_2)
        
    print(target_dir_1+target_files_1[i], target_dir_2+target_files_2[(i + 4) % 8], compare(json_graph_1, json_graph_2))
