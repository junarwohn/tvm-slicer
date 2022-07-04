import socket
import pickle
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

def load_data(data_queue):
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = preprocess(frame)    
        except:
            data_queue.put([])
            # frame_queue.put({-1:-1})
            # result_queue.put({-1:-1})
            break
        data_queue.put(frame)
    print("--------------------Read END--------------------")
    cap.release()

def read_and_inference(data_queue, frame_queue, result_queue):
    model_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
    lib = tvm.runtime.load_module(model_path)

    model_info_path = "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_points[0], args.partition_points[1])
    with open(model_info_path, "r") as json_file:
        model_info = json.load(json_file)

    param_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
    with open(param_path, "rb") as fi:
        loaded_params = bytearray(fi.read())
    
    del model_info['extra'] 

    model = graph_executor.create(json.dumps(model_info), lib, dev)
    model.load_params(loaded_params)

    while True:
        stime= time.time()
        frame = data_queue.get()
        if len(frame) == 0:
            frame_queue.put({-1:-1})
            result_queue.put({-1:-1})
            break

        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

        # ----------------------------
        # # Inference Part
        # ----------------------------
        model.set_input("input_0", input_data)
        model.run()
        outd = model.get_output(0)
        out = outd.numpy().astype(np.float32)
        frame_queue.put({0:frame})
        result_queue.put(out)
        # ----------------------------
        print(time.time() - stime)

# def read_and_inference(frame_queue, result_queue):
#     cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
#     model_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
#     lib = tvm.runtime.load_module(model_path)

#     model_info_path = "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_points[0], args.partition_points[1])
#     with open(model_info_path, "r") as json_file:
#         model_info = json.load(json_file)

#     param_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
#     with open(param_path, "rb") as fi:
#         loaded_params = bytearray(fi.read())
    
#     del model_info['extra'] 

#     model = graph_executor.create(json.dumps(model_info), lib, dev)
#     model.load_params(loaded_params)

#     while (cap.isOpened()):

#         ret, frame = cap.read()
#         try:
#             frame = preprocess(frame)    
#         except:
#             frame_queue.put({-1:-1})
#             result_queue.put({-1:-1})
#             break

#         input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

#         # ----------------------------
#         # # Inference Part
#         # ----------------------------
#         model.set_input("input_0", input_data)
#         model.run()
#         outd = model.get_output(0)
#         out = outd.numpy().astype(np.float32)
#         frame_queue.put({0:frame})
#         result_queue.put(out)
#         # ----------------------------

#     cap.release()

def process_img(frame_queue,result_queue):
        while True:
            # ----------------------------
            # # Visualize Part
            # ----------------------------
            img_in_rgb = frame_queue.get()
            out = result_queue.get()
            if -1 in img_in_rgb.keys():
                while frame_queue.qsize() != 0:
                    frame_queue.get()
                while result_queue.qsize() != 0:
                    result_queue.get()
                break
            th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
            img_in_rgb = img_in_rgb[0]
            img_in_rgb[th == 1] = [0, 0, 255]
            # print(img_in_rgb.flatten()[:10])
            if args.visualize:
                cv2.imshow("received - client", img_in_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # ----------------------------

if __name__ == '__main__':
    print("------------------------")
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.partition_points, sep='')
    data_queue = Queue()
    frame_queue = Queue()
    result_queue = Queue()
    p0 = Process(target=load_data, args=(data_queue,))
    p1 = Process(target=read_and_inference, args=(data_queue, frame_queue, result_queue))
    p2 = Process(target=process_img, args=(frame_queue, result_queue))

    p0.start()
    p1.start() 
    p2.start()
    stime = time.time()
    p0.join()
    p1.join(); 
    p2.join(); 
    print("Total Time :", time.time() - stime)
    print("------------------------")
