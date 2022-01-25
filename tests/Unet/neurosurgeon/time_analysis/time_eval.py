import socket
import pickle
# import cloudpickle as pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
import json
from time import time
import sys
import cv2
import struct
from argparse import ArgumentParser
import ntplib

from tvm.relay.op.transform import repeat 

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 


parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time()
    else:
        return 0


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

model_path = "../src/model/{}_{}_{}_{}.so".format(args.model, args.target, img_size, args.opt_level)
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

cap = cv2.VideoCapture("../src/data/j_scan.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frame", total_frames)
cap.release()
indata = tvm.nd.array(np.random.normal(0,1,(1,3,img_size, img_size)).astype('float32'), device=dev)
print(
    'set_input (ms) :', 
    1000 * model.module.time_evaluator(func_name='set_input', dev=dev, number=total_frames)('input_1', indata).results[0])
print(
    'run (ms) :', 
    1000 * model.module.time_evaluator(func_name='run', dev=dev, number=total_frames)().results[0])
print(
    'get_output (ms) :',
    1000 * model.module.time_evaluator(func_name='get_output', dev=dev, number=total_frames)(0).results[0])
