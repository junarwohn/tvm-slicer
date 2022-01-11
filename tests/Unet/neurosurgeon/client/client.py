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

# Model load


target = 'cuda'
#target = 'llvm'
dev = tvm.cuda(0)
#dev = tvm.cpu(0)
model_path = "../src/model/unet_tvm_front.so"
front_lib = tvm.runtime.load_module(model_path)
front_model = graph_executor.GraphModule(front_lib['default'](dev))

model_info_path = "../src/graph/graph_json_front.json"
with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
output_info = model_info["extra"]["outputs"]

print(input_info, output_info)

print("Model Loaded")

# # Initialize connect

HOST = '192.168.0.184'  
#HOST = '192.168.0.190'
PORT = 9998       
#socket_size = 1 * 1024 * 1024
socket_size = 1024 * 1024

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST, PORT))

print("Connection estabilished")

# Video Load
img_size = 128
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
stime = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (img_size,img_size)) / 255
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    # Execute front
    front_model.set_input("input_0", input_data)
    front_model.run()

    outs = []
    for i, out_idx in enumerate(output_info):
        outs.append([out_idx, front_model.get_output(i).asnumpy().astype(np.float32)])
    
    #print("run finish")
    
    # total_msg = total_num + total_len + idx + len + __obj__ + idx + len + __obj__ ...
    total_msg = struct.pack('i', len(outs))
    objs = []

    # Send msg
    for i, out in outs:
        send_obj = out.tobytes()
        send_obj_len = len(send_obj)
        #print("run", i, send_obj_len, out.shape)
        send_msg = struct.pack('i', i) + struct.pack('i', send_obj_len) + send_obj
        objs.append(send_msg)
        # client_socket.sendall(send_msg)
        # # packet = client_socket.recv(socket_size)
        # if struct.unpack('i', client_socket.recv(4))[0] == 1:
        #     #print("Received")
        #     continue
        # else:
        #     raise Exception("Wrong")

    msg_body = b''
    for o in objs:
        msg_body += o

    total_msg += struct.pack('i', len(msg_body)) + msg_body
    client_socket.sendall(total_msg)

    # send_obj = input_data.tobytes()
    # send_obj_len = len(send_obj)
    # send_msg = struct.pack('i', send_obj_len) + send_obj
    # client_socket.sendall(send_msg)

    # Receive msg
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

    recv_data = np.frombuffer(recv_msg, np.float32).reshape(1,1,img_size,img_size)

    
    #cv2.imshow("original", frame)
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(recv_data.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    #print(np.unique(th, return_counts=True))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

print("Total :", time.time() - stime)
cap.release()
cv2.destroyAllWindows()
