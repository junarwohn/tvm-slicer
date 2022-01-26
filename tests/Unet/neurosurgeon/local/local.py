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
import time
import sys
import cv2
import struct
from argparse import ArgumentParser
import ntplib 

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 


parser = ArgumentParser()
parser.add_argument('--start_point', type=int, default=0)
parser.add_argument('--end_point', type=int, default=-1)
parser.add_argument('--partition_point', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', type=int, default=512, help='set image size')
parser.add_argument('--model', type=str, default='unet', help='name of model')
parser.add_argument('--target', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time.time()
    else:
        return 0


# Model load

target = 'cuda'
#target = 'llvm'
dev = tvm.cuda(0)
#dev = tvm.cpu(0)

model_path = "../src/model/unet_512.so"
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

# Video Load

img_size = 512 
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
stime = time.time()

total_time = 0
total_time_start = time.time()
inference_time = 0
network_time = 0

# TIME_CHECK INIT

time_checker = {
        'READ' : 0,
        'SET_INPUT' : 0,
        'RUN_MODEL': 0,
        'GET_OUTPUT' : 0,
        'PACK' : 0,
        'UNPACK' : 0,
        'VISUALIZE' : 0
}

out = np.zeros((1,1,512,512))

while (cap.isOpened()):
    ### TIME_CHECK : READ
    time_read_start = get_time(args.ntp_enable)
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (img_size, img_size)) / 255
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
    ### TIME_CHECK : READ END
    time_checker['READ'] += get_time(args.ntp_enable) - time_read_start

    ### TIME_CHECK : SET_INPUT
    time_set_input_start = get_time(args.ntp_enable)
 
    # Execute front
    model.set_input("input_1", input_data)
    ### TIME_CHECK : SET_INPUT DONE
    time_checker['SET_INPUT'] += get_time(args.ntp_enable) - time_set_input_start 

    ### TIME_CHECK : RUN_MODEL
    time_run_model_start = get_time(args.ntp_enable)
    model.run()
    outd = model.get_output(0)
    ### TIME_CHECK : RUN_MODEL DONE
    time_checker['RUN_MODEL'] += get_time(args.ntp_enable) - time_run_model_start 

    ### TIME_CHECK : VISUALIZE 
    time_visualize_start = get_time(args.ntp_enable)
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #ret, frame = cap.read()
    ### TIME_CHECK : VISUALIZE END
    time_checker['VISUALIZE'] += get_time(args.ntp_enable) - time_visualize_start


    ### TIME_CHECK : GET_OUTPUT
    time_get_output_start = get_time(args.ntp_enable)
    #out = model.get_output(0).asnumpy().astype(np.float32)
    #out = model.get_output(0)
    out = outd.numpy().astype(np.float32)
    ### TIME_CHECK : GET_OUTPUT
    time_checker['GET_OUTPUT'] += get_time(args.ntp_enable) - time_get_output_start
    #out = out.numpy().astype(np.float32)

    #### TIME_CHECK : VISUALIZE 
    #time_visualize_start = get_time(args.ntp_enable)
    #img_in_rgb = frame
    #th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    #img_in_rgb[th == 1] = [0, 0, 255]
    #cv2.imshow("received - client", img_in_rgb)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    ##ret, frame = cap.read()
    #### TIME_CHECK : VISUALIZE END
    #time_checker['VISUALIZE'] += get_time(args.ntp_enable) - time_visualize_start

print("Total time :", time.time() - stime)

total_time_checker = 0
total_inferenece_checker = 0
for key in time_checker:
    print(key, ':', time_checker[key] / 352 * 1000)
    total_time_checker += time_checker[key]
    if key == 'SET_INPUT' or key == 'RUN' or key == 'GET_OUTPUT':
        total_inferenece_checker += time_checker[key]
print("total_time_checker :", total_time_checker)
print("total_inferenece_checker :", total_inferenece_checker / 352 * 1000)


cap.release()
