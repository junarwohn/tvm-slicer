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
parser.add_argument('--ntp_enable', type=int, default=1, help='ntp support')

args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time.time()
    else:
        return 0

# Model load
if args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
else:
    raise Exception("Wrong device")

model_path = "../src/model/{}_{}_front_{}_{}.so".format(args.model, args.target, args.img_size, args.partition_point)
front_lib = tvm.runtime.load_module(model_path)
front_model = graph_executor.GraphModule(front_lib['default'](dev))

model_info_path = "../src/graph/{}_{}_front_{}_{}.json".format(args.model, args.target, args.img_size, args.partition_point)
with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
output_info = model_info["extra"]["outputs"]

#print(input_info, output_info)

#print("Model Loaded")

# # Initialize connect

HOST_IP = args.ip
PORT = 9998       
#socket_size = 16 * 1024 * 1024
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))

#client_socket.sendall(struct.pack('?', True))
#base_time = time.time()
#print(base_time)

#print("Connection estabilished")

# Video Load
img_size = 512 
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
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

while (cap.isOpened()):
    ### TIME_CHECK : READ
    time_read_start = get_time(args.ntp_enable)
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (img_size,img_size)) / 255
    except:
        #print("Transmission End")
        time_sent = struct.pack('d', get_time(args.ntp_enable))
        total_msg = struct.pack('i', 0)
        client_socket.sendall(time_sent + total_msg)
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
    ### TIME_CHECK : READ END
    time_checker['READ'] += get_time(args.ntp_enable) - time_read_start

    inference_time_start = time.time()

    # Execute front
    ### TIME_CHECK : SET_INPUT
    time_set_input_start = get_time(args.ntp_enable)
    front_model.set_input("input_0", input_data)
    ### TIME_CHECK : SET_INPUT DONE
    time_checker['SET_INPUT'] += get_time(args.ntp_enable) - time_set_input_start 

    ### TIME_CHECK : RUN_MODEL
    time_run_model_start = get_time(args.ntp_enable)
    front_model.run()
    ### TIME_CHECK : RUN_MODEL DONE
    time_checker['RUN_MODEL'] += get_time(args.ntp_enable) - time_run_model_start 

    ### TIME_CHECK : GET_OUTPUT
    time_get_output_start = get_time(args.ntp_enable)
    outs = []
    for i, out_idx in enumerate(output_info):
        outs.append([out_idx, front_model.get_output(i).asnumpy().astype(np.float32)])
    ### TIME_CHECK : GET_OUTPUT
    time_checker['GET_OUTPUT'] += get_time(args.ntp_enable) - time_get_output_start

    inference_time += time.time() - inference_time_start
    
    ### TIME_CHECK : PACK
    time_pack_start = get_time(args.ntp_enable)
    time_sent = struct.pack('d', get_time(args.ntp_enable))
    total_msg = struct.pack('i', len(outs))
    objs = []

    # Send msg
    for i, out in outs:
        send_obj = out.tobytes()
        send_obj_len = len(send_obj)
        #print("run", i, send_obj_len, out.shape)
        send_msg = struct.pack('i', i) + struct.pack('i', send_obj_len) + send_obj
        objs.append(send_msg)

    msg_body = b''
    for o in objs:
        msg_body += o

    total_msg += struct.pack('i', len(msg_body)) + msg_body
    ### TIME_CHECK : PACK DONE
    time_checker['PACK'] += get_time(args.ntp_enable) - time_pack_start

    client_socket.sendall(time_sent + total_msg)

    time_sent = struct.unpack('d', client_socket.recv(8))[0]

    recv_msg_idx = struct.unpack('i', client_socket.recv(4))[0]
    recv_msg_len = struct.unpack('i', client_socket.recv(4))[0]
    if recv_msg_len == 0:
        break
    
    packet = client_socket.recv(socket_size)
    # packet = client_socket.recv()
    recv_msg = packet
    while len(recv_msg) < recv_msg_len:
        packet = client_socket.recv(socket_size)
        # packet = client_socket.recv()
        recv_msg += packet

    ### TIME_CHECK : UNPACK 
    time_unpack_start = get_time(args.ntp_enable)

    recv_data = np.frombuffer(recv_msg, np.float32).reshape(1,1,img_size,img_size)
    ### TIME_CHECK : UNPACK END
    time_checker['UNPACK'] += get_time(args.ntp_enable) - time_unpack_start

    network_time += get_time(args.ntp_enable) - time_sent 
    #network_time += time_sent 

    
    ### TIME_CHECK : VISUALIZE 
    time_visualize_start = get_time(args.ntp_enable)
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(recv_data.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    #cv2.imshow("original", frame)
    #print(np.unique(th, return_counts=True))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    ### TIME_CHECK : VISUALIZE END
    time_checker['VISUALIZE'] += get_time(args.ntp_enable) - time_visualize_start

total_time = time.time() - total_time_start
print("total time :", total_time)
print("inference time :", inference_time)
print("network time :", network_time)
print("number of input :", len(input_info))
print("index of input :", input_info[0])
print("number of output :", len(output_info))
print("index of output :", output_info[0])
print("data input size :", len(input_data.tobytes()))
print("data receive size :", recv_msg_len)
print("data send size :", send_obj_len)
total_time_checker = 0
for key in time_checker:
    print(key, ':', time_checker[key])
    total_time_checker += time_checker[key]
print("total_time_checker :", total_time_checker)

cap.release()
cv2.destroyAllWindows()
