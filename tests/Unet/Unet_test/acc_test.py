import cv2
import os
from argparse import ArgumentParser
import tvm
from tvm.contrib import graph_executor
import json
import time
import numpy as np

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--front', '-f', nargs='+', type=str, default=[], help='set front partition point')
parser.add_argument('--back', '-b', nargs='+', type=str, default=[], help='set back partition point')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
args = parser.parse_args()

base_path = os.environ.get("TS_DATA_PATH", "")
raw_data_path = base_path + "/us-bmod/{}/raw/0/"
mask_data_path = base_path + "/us-bmod/{}/mask/0/"

model_config = args.model_config
quantization_level = args.quantization_level


def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            # return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)).astype(np.float32) / 255
            return cv2.resize(img, (im_sz,im_sz)).astype(np.float32) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

preprocess = make_preprocess(args.model, args.img_size)

def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

    # If there is no model to be executed
    if len(partition_points) == 1:
        return [partition_points], [partition_points], []

    # Load front model json infos
    for i in range(len(partition_points) - 1):
        # start_point = partition_points[i]
        # end_point = partition_points[i + 1]
        # current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        start_points = [int(i) for i in partition_points[i].split(',')]
        end_points =  [int(i) for i in partition_points[i + 1].split(',')]

        # with open(current_file_path + "unet_as_{}_{}_{}_{}_{}-{}.json".format(*model_config, start_point, end_point), "r") as json_file:
        with open("UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json".format(
            *model_config, 
            quantization_level, 
            "_".join(map(str, start_points)), 
            "_".join(map(str, end_points))), "r") as json_file:

            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))

    return model_input_indexs, model_output_indexs, model_graph_json_strs


# target and dev set
if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()


# base_path = raw_data_path.format('validation')
base_path = raw_data_path.format('test')
raw_file = [base_path +i for i in sorted(os.listdir(base_path))]


# base_path = mask_data_path.format('validation')
base_path = mask_data_path.format('test')
mask_file = [base_path +i for i in sorted(os.listdir(base_path))]


def load_data():
    data_queue = []
    mask_queue = []
    for r, m in zip(raw_file, mask_file):
        ri = cv2.imread(r)
        ri = preprocess(ri)    

        mi = cv2.imread(m, 0)
        mi = preprocess(mi)    

        data_queue.append(ri)
        mask_queue.append(mi)
    return data_queue, mask_queue

def IoU(M1, M2):
    M1 = M1.flatten()
    M1 = np.where(M1 < 0.5, False, True)

    M2 = M2.flatten()
    M2 = np.where(M2 < 0.5, False, True)

    combine = np.logical_or(M1,M2)
    overlap = np.logical_and(M1,M2)
    if M1.sum() == 0:
        return -1
    return overlap.sum() / combine.sum() 

if __name__ == '__main__':
    print("------------------------")
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.front, args.back, sep='')

    # Load model
    points_front_model = args.partition_points

    front_input_idxs, front_output_idxs, front_graph_json_strs = get_model_info(points_front_model)

    # Load models
    model_path = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so".format(*model_config, quantization_level)
    # model_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
    lib = tvm.runtime.load_module(model_path)

    param_path = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params".format(*model_config, quantization_level)
    with open(param_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    front_models = []
    for graph_json_str in front_graph_json_strs:
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        front_models.append(model)

    in_data = {0 : 0}
    # Load network connection

    data_queue, mask_queue = load_data()

    stime = time.time()
    cnt = 0
    result = 0
    for frame, mframe in zip(data_queue, mask_queue):
        in_data = {}

        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        in_data[0] = input_data

        # Front inference
        for in_indexs, out_indexs, model in zip(front_input_idxs, front_output_idxs, front_models):
            # set input
            for input_index in in_indexs:
                model.set_input("input_{}".format(input_index), in_data[input_index])
            # run model
            model.run()
            for i, output_index in enumerate(out_indexs):
                in_data[output_index] = model.get_output(i).numpy()

        out = in_data[output_index]

        img_in_rgb = frame

        # img_in_rgb[mframe == 1] = [0, 0, 255]
        # cv2.imshow("received - client", img_in_rgb)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        # img_in_rgb = cv2.cvtColor(mframe,cv2.COLOR_GRAY2RGB)

        # th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (256,256))
        # img_in_rgb[th == 1] = [0, 0, 255]
        # cv2.imshow("received - client", img_in_rgb)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        # Option : visualize

        iou = IoU(out, mframe)
        if iou != -1:
            result += iou
            cnt += 1
    print(result / cnt)
    # print(time.time() - stime)

    print("------------------------")
