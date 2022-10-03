import numpy as np
import tvm
from tvm.contrib import graph_executor
from argparse import ArgumentParser
import json
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

if is_jetson == 1:
    target = tvm.target.Target("nvidia/jetson-nano")
    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_53"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
    assert target.attrs["registers_per_block"] == 32768
    dev = tvm.cuda()
else:
    if args.target == 'llvm':
        target = 'llvm'
        dev = tvm.cpu()
    elif args.target == 'cuda':
        target = 'cuda'
        dev = tvm.cuda()
    elif args.target == 'opencl':
        target = 'opencl'
        dev = tvm.opencl()


def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []
    model_dummy_inputs = []
    # If there is no model to be executed
    if len(partition_points) == 1:
        partition_points = list(map(int, partition_points))
        return [partition_points], [partition_points], []

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

model_path = model_format.format(*model_config, quantization_level)
lib = tvm.runtime.load_module(model_path)

if is_jetson == 1:
    param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.params"
else:
    param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params"

param_path = param_format.format(*model_config, quantization_level)
with open(param_path, "rb") as fi:
    loaded_params = bytearray(fi.read())

input_idxs, output_idxs, graph_json_strs, dummy_inputs = get_model_info(args.partition_points)
slice_points = [[args.partition_points[i],args.partition_points[i+1]] for i in range(len(args.partition_points)-1)]
print(*args.partition_points)
for input_idx, output_idx, graph_json_str, dummy_input, slice_point in zip(input_idxs, output_idxs, graph_json_strs, dummy_inputs, slice_points):
    set_input_time = 0
    run_time = 0
    
    model = graph_executor.create(graph_json_str, lib, dev)
    model.load_params(loaded_params)

    total_frames = 100
    # print(input_idx, len(dummy_input))
    for ii, di in zip(input_idx,dummy_input):
        indata = tvm.nd.array(di, device=dev)
        set_input_time += model.module.time_evaluator(func_name='set_input', dev=dev, number=total_frames)('input_{}'.format(ii), indata).results[0]
    run_time += model.module.time_evaluator(func_name='run', dev=dev, number=total_frames)().results[0]
    
    print(*slice_point)
    print(f'{set_input_time:.10f}, {run_time:.10f}')
