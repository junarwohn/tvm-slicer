from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
args = parser.parse_args()

model_config = args.model_config
quantization_level = args.quantization_level
is_jetson = args.jetson
img_size = args.img_size

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
