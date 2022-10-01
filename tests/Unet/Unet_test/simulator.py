import json
from argparse import ArgumentParser
import numpy as np
import pickle


parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
args = parser.parse_args()

model_config = args.model_config
quantization_level = args.quantization_level
is_jetson = args.jetson
img_size = args.img_size

def network_simulator(data_size, network='lan'):
    lan_weight = 0.000000008501843269
    lan_bias = -0.000009457928473
    
    wifi_weight = 0.0000000314463409
    wifi_bias = -0.003767568519
    
    if network == 'lan':
        result_time = lan_weight * data_size + lan_bias
        if result_time < 0:
            result_time = 0
            
    else:
        result_time = wifi_weight * data_size + wifi_bias
        if result_time < 0.0005:
            result_time = 0.0005
            
    return result_time
    
    
def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []
    model_dummy_inputs = []
    # If there is no model to be executed
    if len(partition_points) == 1:
        partition_points = list(map(int, partition_points))
        return [partition_points], [partition_points], []
    
    # False -> True -> False
    is_time_for_jetson = False

    # Load front model json infos
    for i in range(len(partition_points) - 1):
        # start_point = partition_points[i]
        # end_point = partition_points[i + 1]
        start_points = [int(i) for i in partition_points[i].split(',')]
        end_points =  [int(i) for i in partition_points[i + 1].split(',')]
        # print(start_points, end_points)
        # current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        target_dir = "./slice_config/M[{}-{}-{}-{}]_Q[{}]_S[{}-{}-{}-{}]/".format(*model_config, quantization_level, *list(map(int, args.partition_points)))
        json_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json"

        with open(target_dir + json_format.format(
            *model_config, 
            quantization_level, 
            "_".join(map(str,[i for i in start_points])), 
            "_".join(map(str, end_points))), "r") as json_file:
        # with open("unet_as_{}_{}_{}_{}_{}-{}.json".format(*model_config, start_point, end_point), "r") as json_file:
            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))
        input_node_indexs = [i for i, node in enumerate(graph_json['nodes']) if 'input' in node['name']]
        input_types = [graph_json['attrs']['dltype'][1][i] for i in input_node_indexs]
        input_shapes = [graph_json['attrs']['shape'][1][i] for i in input_node_indexs]
        dummy = []
        for dt, sh in zip(input_types, input_shapes):
            dummy.append(np.random.normal(0, 1, sh).astype(dt))
        model_dummy_inputs.append(dummy)
    return model_input_indexs, model_output_indexs, model_graph_json_strs, model_dummy_inputs


# Load models
if is_jetson == 1:
    model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.so"
else:
    model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so"

if is_jetson == 1:
    param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.params"
else:
    param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params"

input_idxs, output_idxs, graph_json_strs, dummy_inputs = get_model_info(args.partition_points)
slice_points = [[args.partition_points[i],args.partition_points[i+1]] for i in range(len(args.partition_points)-1)]
# writedata.py
time_2080ti = open("./time_2080ti.txt", 'r')
time_2080ti_info = []
while True:
    line = time_2080ti.readline()
    if not line: break
    if "{} {} {} {}".format(*args.partition_points) in line:
        for i in range(3):
            time_2080ti.readline().split('\n')[0]
            time_2080ti_info.append(time_2080ti.readline().split('\n')[0].split(','))
time_2080ti.close()

time_jetson = open("./time_jetson.txt", 'r')
time_jetson_info = []
while True:
    line = time_jetson.readline()
    if not line: break
    if "{} {} {} {}".format(*args.partition_points) in line:
        for i in range(3):
            time_jetson.readline().split('\n')[0]
            time_jetson_info.append(time_jetson.readline().split('\n')[0].split(','))
time_jetson.close()


if len(input_idxs) == 3:
    front_input_idxs, front_output_idxs, front_graph_json_strs, front_dummy_inputs = get_model_info(args.partition_points[:2])
    server_input_idxs, server_output_idxs, _, server_dummy_inputs= get_model_info(args.partition_points[1:3])
    back_input_idxs, back_output_idxs, back_graph_json_strs, back_dummy_inputs = get_model_info(args.partition_points[2:4])

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
    pass_queue_idxs = np.intersect1d(total_front_output_idxs, total_back_input_idxs)
    recv_queue_idxs = np.intersect1d(total_server_output_idxs, total_back_input_idxs)
    # print(send_queue_idxs, pass_queue_idxs, recv_queue_idxs)
    # print(front_input_idxs, server_input_idxs, back_input_idxs)
    
    total_network_time = 0
    total_inference_time = 0
    ### TIME
    inputs = list(front_input_idxs[0]) + list(server_input_idxs[0]) + list(back_input_idxs[0])
    input_dummys = list(front_dummy_inputs[0]) + list(server_dummy_inputs[0]) + list(back_dummy_inputs[0])
    for data_idx in list(send_queue_idxs) + list(recv_queue_idxs):
        idx = inputs.index(data_idx)
        msg_size = len(pickle.dumps(input_dummys[idx]))
        # print(network_simulator(msg_size, 'wifi'))
        total_network_time += network_simulator(msg_size, 'wifi')

    ### INFERENCE
    total_inference_time += sum(list(map(float, time_jetson_info[0]))) + sum(list(map(float, time_2080ti_info[1]))) + sum(list(map(float, time_jetson_info[2])))
    
else:
    print("Wrong slicing configure!!!!!")

print(*args.partition_points, end=" : ")
print(total_network_time + total_inference_time)