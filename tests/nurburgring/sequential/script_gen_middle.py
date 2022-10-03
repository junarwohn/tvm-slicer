from cv2 import CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION
from grpc import server
import numpy as np
from argparse import ArgumentParser
from itertools import combinations
import os
import json
import itertools

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--front', '-f', nargs='+', type=int, default=0, help='set front point')
parser.add_argument('--back', '-b', nargs='+', type=int, default=0, help='set back point')
parser.add_argument('--name', '-n', type=str, default='client', help='set program name')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()

def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

    # If there is no model to be executed
    if len(partition_points) == 1:
        return [partition_points], [partition_points], []

    # Load front model json infos
    for i in range(len(partition_points) - 1):
        start_point = partition_points[i]
        end_point = partition_points[i + 1]
        current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))

    return model_input_indexs, model_output_indexs, model_graph_json_strs

partition_points = args.partition_points

client_side = []
server_side = []

if args.name == 'client':
    print("rm client_log.txt")

elif args.name == 'server':
    print("rm server_log.txt")

_,_,graph_json_raw = get_model_info(partition_points)
graph = json.loads(graph_json_raw[0])
candidates_points = []
for idx, node in enumerate(graph['nodes']):
    inputs = [i[0] for i in node['inputs']]
    dtype = graph['attrs']['dltype'][1][idx]
    if dtype == 'int8':
        # print(idx, node['name'], dtype)
        candidates_points.append(idx)

c1 = [[args.partition_points[0]] + list(i) + [args.partition_points[-1]] for  i in itertools.combinations(candidates_points, 2)]
c2 = [[args.partition_points[0]] + list(i) + [args.partition_points[-1]] for  i in itertools.combinations(candidates_points, 1)]
candidates = c1 + c2
tmp = []
for c in candidates:
    is_too_narrow = False
    for i in range(len(c) - 1):
        if c[i+1] - c[i] == 1:
            is_too_narrow = True
            break
    if not is_too_narrow:
        tmp.append(c)
candidates = tmp

for candi in candidates:
    if len(candi) == 4:
        # 0-1 : 1-2 : 2-3
        if args.name == 'client':
            print("python3 client.py", "-f", candi[0], candi[1], "-b", candi[2], candi[3])
            print("sleep 3")
            print("python3 client.py", "-f", candi[0], candi[1], "-b", candi[2], candi[3], ">> client_log.txt")
            print("sleep 3")
        else:
            print("python3 server.py -p", candi[1], candi[2])
            print("python3 server.py -p", candi[1], candi[2], ">> server_log.txt")
    elif len(candi) == 3:
        # 0-1 : 1-2 : NULL
        # NULL : 0-1 : 1-2
        if args.name == 'client':
            print("python3 client.py", "-f", candi[0], candi[1], "-b", candi[2])
            print("sleep 3")
            print("python3 client.py", "-f", candi[0], candi[1], "-b", candi[2], ">> server_log.txt")
            print("sleep 3")

            print("python3 server.py", "-f", candi[0], "-b", candi[1], candi[2])
            print("sleep 3")
            print("python3 server.py", "-f", candi[0], "-b", candi[1], candi[2], ">> server_log.txt")
            print("sleep 3")

        else:
            print("python3 server.py -p", candi[1], candi[2])
            print("python3 server.py -p", candi[1], candi[2], ">> server_log.txt")

            print("python3 server.py -p", candi[0], candi[1])
            print("python3 server.py -p", candi[0], candi[1], ">> server_log.txt")