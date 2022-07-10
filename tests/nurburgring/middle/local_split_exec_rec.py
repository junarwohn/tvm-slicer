import socket
import pickle
from statistics import mode
from unittest import result
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
from multiprocessing import Process, Queue
import os

""" Test local tvm execution and measure the performance. """

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 

# Argument Parser
parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='input ip of host')
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
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

img_size = args.img_size 


def read_and_inference():
    result = []
    # ----------------------------
    # # load data
    # ----------------------------
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    # ----------------------------


    # ----------------------------
    # # load model
    # ----------------------------
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
        print(start_point, end_point)
        with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        # Temp last model bug fix
        if len(output_indexs) == 0:
            output_indexs = [graph_json['heads'][0][0]]
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))


    param_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
    with open(param_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    models = []

    # print("Before dummpy executor")
    # for i in range(5):
    #     print("Wait....", i)
    #     time.sleep(1)
     
    # model = graph_executor.create(model_graph_json_strs[0], lib, dev)
    # model.load_params(loaded_params)
    # l = model   

    for graph_json_str in model_graph_json_strs:
        # unique_storage_id = np.unique(json.loads(graph_json_str)['attrs']['storage_id'][1])
        # print(len(unique_storage_id))
        print("Before create executor")
        for i in range(5):
            print("Wait....", i)
            time.sleep(1)
        model = graph_executor.create(graph_json_str, lib, dev)
        print("Load params....")
        time.sleep(5)
        model.load_params(loaded_params)
        models.append(model)
        
    in_data = {0 : 0}
    # ----------------------------
    time.sleep(5)
    timer_set_input = {
        model_index : {
            input_index : 0 for input_index in model_info 
        } 
        for model_index, model_info in enumerate(model_input_indexs)
    }

    timer_run_model = {
        model_index : 0 for model_index in range(len(models))
    }

    timer_get_output = {
        model_index : {
            output_index : 0 for output_index in range(len(model_info))
        } 
        for model_index, model_info in enumerate(model_output_indexs)
    }

    while (cap.isOpened()):

        ret, frame = cap.read()
        try:
            frame = preprocess(frame)    
        except:
            break

        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        in_data[0] = input_data

        for model_index, [in_indexs, out_indexs, model] in enumerate(zip(model_input_indexs, model_output_indexs, models)):
            # set input
            for input_index in in_indexs:
                # timer start
                time_start = time.time()
                model.set_input("input_{}".format(input_index), in_data[input_index])
                # timer end
                timer_set_input[model_index][input_index] += time.time() - time_start
            
            # run model
            # timer start
            time_start = time.time()
            model.run()
            dev.sync()
            # timer end
            timer_run_model[model_index] += time.time() - time_start

            # get output
            for i, output_index in enumerate(out_indexs):
                time_start = time.time()
                in_data[output_index] = model.get_output(i).numpy()
                pre_output = in_data[output_index]
                timer_get_output[model_index][i] += time.time() - time_start

        # ----------------------------
        # # Visualize Part
        # ----------------------------
        img_in_rgb = frame
        out = pre_output
        th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        img_in_rgb[th == 1] = [0, 0, 255]
        if args.visualize:
            cv2.imshow("received - client", img_in_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # ----------------------------
        result.append(th)
    cap.release()
    np.save("./split.npy", np.array(result))
    # ----------------------------
    # # Print infos
    # ----------------------------
    print("timer_set_input")
    for model_index in timer_set_input:
        print(model_index, ": ", end='')
        for input_index in timer_set_input[model_index]:
            print('[', input_index, ']', "-", timer_set_input[model_index][input_index], ",",sep="")
    sum_set_input = 0
    for model_index in timer_set_input:
        for input_index in timer_set_input[model_index]:
            sum_set_input += timer_set_input[model_index][input_index]
    print("sum of set_input", sum_set_input) 
    print()
    
    print("timer_run_model")
    for model_index in timer_run_model:
        print(model_index, " : ", timer_run_model[model_index], ",",sep="")
    sum_run_model = 0
    for model_index in timer_run_model:
        sum_run_model += timer_run_model[model_index]
    print("sum of run_model", sum_run_model) 
    print()
    
    print("timer_get_output")
    for model_index in timer_get_output:
        print(model_index, ": ", end='')
        for output_index in timer_get_output[model_index]:
            print('[', output_index, ']', "-", timer_get_output[model_index][output_index], ",",sep="")
    sum_get_input = 0
    for model_index in timer_get_output:
        for output_index in timer_get_output[model_index]:
            sum_get_input += timer_get_output[model_index][output_index]
    print("sum of set_input", sum_get_input) 
    print()

    print("sum of inference", sum_set_input + sum_run_model + sum_get_input)
    # ----------------------------

if __name__ == '__main__':
    print("------------------------")
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.partition_points, sep='')
    p1 = Process(target=read_and_inference)

    p1.start() 
    stime = time.time()
    p1.join(); 
    print("Total Time :", time.time() - stime)
    print("------------------------")
