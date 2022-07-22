from argparse import ArgumentParser
import os
import json
import re
import numpy as np
from sklearn import datasets

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--client', '-c' , type=str, default='model_test_log_jetson.txt', help='type target file')
parser.add_argument('--server', '-s' , type=str, default='model_test_log_2080ti_300.txt', help='type target file')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
args = parser.parse_args()


NETWORK_WEIGHT = 0.00000000854636931
NETWORK_BAIS = 0.000004568534259
MODEL_TEST_TIME = 253
NETWORK_TEST_TIME = 1000
def network_cost(data_shape, data_type):
    data_size = 1
    for d in data_shape:
        data_size = data_size * d
    if data_type == 'float32':
        return 4 * data_size * NETWORK_WEIGHT + NETWORK_BAIS 
    elif data_type == 'int8':
        return data_size * NETWORK_WEIGHT + NETWORK_BAIS 
    else:
        return None

current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

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

# lookuptable : { str(src) : [str(dst1), str(dst2) ...] }
lut_src_dst_client = dict()
# lookuptable : { str(src-dst) : time }
lut_path_time_client = dict()

f = open("./" + args.client, 'r')
# path_pat = "\[\d+, \d+\]"
# data format
while True:
    line = f.readline()
    line = line.strip()
    if not line:
        break
    src, dst = [int(i) for i in re.findall(r"\d+", re.findall(r"\[\d+, \d+\]", line)[0])]
    time = float(re.findall(r"\d+\.\d+", line)[0])
    if src in lut_src_dst_client:
        lut_src_dst_client[src].append(dst)
    else:
        lut_src_dst_client[src] = [dst]
    
    lut_path_time_client["{}_{}".format(src, dst)] = time/MODEL_TEST_TIME

# sort lut
for key in lut_src_dst_client:
    lut_src_dst_client[key] = sorted(lut_src_dst_client[key])


# lookuptable : { str(src) : [str(dst1), str(dst2) ...] }
lut_src_dst_server = dict()
# lookuptable : { str(src-dst) : time }
lut_path_time_server = dict()
f.close()

f = open("./" + args.server, 'r')
# path_pat = "\[\d+, \d+\]"
# data format
while True:
    line = f.readline()
    line = line.strip()
    if not line:
        break
    src, dst = [int(i) for i in re.findall(r"\d+", re.findall(r"\[\d+, \d+\]", line)[0])]
    time = float(re.findall(r"\d+\.\d+", line)[0])
    if src in lut_src_dst_server:
        lut_src_dst_server[src].append(dst)
    else:
        lut_src_dst_server[src] = [dst]
    
    lut_path_time_server["{}_{}".format(src, dst)] = time/MODEL_TEST_TIME

# sort lut
for key in lut_src_dst_server:
    lut_src_dst_server[key] = sorted(lut_src_dst_server[key])
            
f.close()
            


# check total time
start_node = 0
end_node = lut_src_dst_client[0][-1]
_, _, total_graph_json = get_model_info([start_node, end_node])
total_graph_json = json.loads(total_graph_json[0])

def dfs(history, end_node, lut, cost_lut_client, cost_lut_server):
    last_visit = history[-1]
    if len(history) == 3 and last_visit != end_node:
        history += [lut[last_visit][-1]]
        last_visit = history[-1]
    if last_visit == end_node:
        # Combination
        # len == 2 => client only
        if len(history) == 2:
            cost = sum([cost_lut_client["{}_{}".format(history[i], history[i+1])] for i in range(len(history) - 1)])
            print(history, cost, "(client)")
        # len == 3 => client->server result, server->client result
        elif len(history) == 3:
            # client->server
            _, front_output_idxs, _ = get_model_info(history[0:2])
            server_input_idxs, server_output_idxs, _ = get_model_info(history[1:3])

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_input_idxs = []
            for i in server_input_idxs:
                total_server_input_idxs += i

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_output_idxs = []
            for i in server_output_idxs:
                total_server_output_idxs += i

            send_queue_idxs = total_server_input_idxs
            recv_queue_idxs = [end_node]

            front_inference_cost = cost_lut_client["{}_{}".format(*history[0:2])]
            front2server_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in send_queue_idxs])
            server_inference_cost = cost_lut_server["{}_{}".format(*history[1:3])]
            server2client_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in recv_queue_idxs])
            total_cost = front_inference_cost + front2server_cost + server_inference_cost + server2client_cost
            print(history, total_cost, "(client->server)")
            # print(front2server_cost + server2client_cost)


            # server->client
            _, front_output_idxs, _ = get_model_info(history[0:2])
            server_input_idxs, server_output_idxs, _ = get_model_info(history[1:3])

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_input_idxs = []
            for i in server_input_idxs:
                total_server_input_idxs += i

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_output_idxs = []
            for i in server_output_idxs:
                total_server_output_idxs += i

            send_queue_idxs = [0]
            recv_queue_idxs = total_server_output_idxs

            front2server_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in send_queue_idxs])
            server_inference_cost = cost_lut_server["{}_{}".format(*history[0:2])]
            server2client_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in recv_queue_idxs])
            back_inference_cost = cost_lut_client["{}_{}".format(*history[1:3])]
            total_cost = front2server_cost + server_inference_cost + server2client_cost + back_inference_cost
            print(history, total_cost, "(server->client)")
            # print(front2server_cost + server2client_cost)
        elif len(history) == 4:
            _, front_output_idxs, _ = get_model_info(history[0:2])
            server_input_idxs, server_output_idxs, _ = get_model_info(history[1:3])
            back_input_idxs, _, _ = get_model_info(history[2:4])

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_input_idxs = []
            for i in server_input_idxs:
                total_server_input_idxs += i

            total_back_input_idxs = []
            for i in back_input_idxs:
                total_back_input_idxs += i

            total_front_output_idxs = []
            for i in front_output_idxs:
                total_front_output_idxs += i

            total_server_output_idxs = []
            for i in server_output_idxs:
                total_server_output_idxs += i

            total_back_input_idxs = []
            for i in back_input_idxs:
                total_back_input_idxs += i

            send_queue_idxs = total_server_input_idxs
            recv_queue_idxs = np.intersect1d(total_server_output_idxs, total_back_input_idxs)
            
            front_inference_cost = cost_lut_client["{}_{}".format(*history[0:2])]
            front2server_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in send_queue_idxs])
            server_inference_cost = cost_lut_server["{}_{}".format(*history[1:3])]
            server2client_cost = sum([network_cost(total_graph_json["attrs"]["shape"][1][idx], total_graph_json["attrs"]['dltype'][1][idx]) for idx in recv_queue_idxs])
            back_inference_cost = cost_lut_client["{}_{}".format(*history[2:4])]
            total_cost = front_inference_cost + front2server_cost + server_inference_cost + server2client_cost + back_inference_cost
            print(history, total_cost, "(client->server->client)")
            # print(front2server_cost + server2client_cost)

    else:
        for dst in lut[last_visit]:
            dfs(history+[dst], end_node, lut, cost_lut_client, cost_lut_server)

dfs([start_node], end_node, lut_src_dst_server, lut_path_time_client, lut_path_time_server)

