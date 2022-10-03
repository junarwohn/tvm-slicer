import socket
import pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
import json
import time
import sys
import cv2
import struct
from argparse import ArgumentParser
import ntplib 
import os

""" Test local tvm execution and measure the performance. """

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 

# Argument Parser
parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--add_quantize_layer', '-q', type=int, default=0, help='add int8 quantize layer at sliced edge')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time.time()
    else:
        return 0

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

preprocess = make_preprocess(args.model, args.img_size)

# Model load

if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()

# Load models
model_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
lib = tvm.runtime.load_module(model_path)
partition_points = args.partition_points
current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

model_input_indexs = []
model_output_indexs = []
model_graph_json_strs = []

for i in range(len(partition_points) - 1):
    start_point = partition_points[i]
    end_point = partition_points[i + 1]
    with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
        graph_json = json.load(json_file)
    input_indexs = graph_json['extra']["inputs"]
    output_indexs = graph_json['extra']["outputs"]
    
    model_input_indexs.append(input_indexs)
    model_output_indexs.append(output_indexs)
    del graph_json['extra']
    model_graph_json_strs.append(json.dumps(graph_json))


param_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
with open(param_path, "rb") as fi:
    loaded_params = bytearray(fi.read())

models = []
for graph_json_str in model_graph_json_strs:
    model = graph_executor.create(graph_json_str, lib, dev)
    model.load_params(loaded_params)
    models.append(model)

# Video Load
img_size = 512 
cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
stime = time.time()

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0
total_recv_msg_size = 0
total_send_msg_size = 0

# Loop starts
timer_toal_start = time.time()

in_data = {0 : 0}

while (cap.isOpened()):
    
    ret, frame = cap.read()
    try:
        frame = preprocess(frame)    
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    in_data[0] = input_data
    out_data = []
    for in_indexs, out_indexs, model in zip(model_input_indexs, model_output_indexs, models):
        # set input
        for input_index in in_indexs:
            model.set_input("input_{}".format(input_index), in_data[input_index])
        
        # run model
        model.run()

        # get output
        for i, output_index in enumerate(out_indexs):
            in_data[output_index] = model.get_output(i)

    # Check last output index of model     
    if len(model_output_indexs[-1]) == 1:
        out = in_data[model_output_indexs[-1][0]].numpy()
        # print(out)
    else:
        print("Wrong output of last model")

    # ----------------------------
    # # Visualize Part
    # ----------------------------
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    img_in_rgb[th == 1] = [0, 0, 255]
    if args.visualize:
        cv2.imshow("received - client", img_in_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # ----------------------------

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network

print("----------------------------")
print("Execution Result ")
print("model:[{}], target:[{}], img_size:[{}], opt_level:[{}]".format(args.model, args.target, args.img_size, args.opt_level))
print("total time :", timer_total)
print("inference time :", timer_inference)
print("----------------------------")

cap.release()