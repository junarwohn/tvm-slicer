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
dev = tvm.cuda(0)
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
PORT = 9998       
socket_size = 1 * 1024 * 1024

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST, PORT))

print("Connection estabilished")

# Video Load

cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame[490:1800, 900:2850], (512,512)) / 255
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    # Execute front
    front_model.set_input("input_0", input_data)
    front_model.run()

    outs = []
    for i, out_idx in enumerate(output_info):
        outs.append([out_idx, front_model.get_output(i).asnumpy().astype(np.float32)])
    
    print("run finish")
    
    # Send msg
    for i, out in outs:
        send_obj = out.tobytes()
        send_obj_len = len(send_obj)
        print("run", i, send_obj_len, out.shape)
        send_msg = struct.pack('i', i) + struct.pack('i', send_obj_len) + send_obj
        client_socket.sendall(send_msg)
        # packet = client_socket.recv(socket_size)
        if struct.unpack('i', client_socket.recv(4))[0] == 1:
            print("Received")
            continue
        else:
            raise Exception("Wrong")
        

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

    recv_data = np.frombuffer(recv_msg, np.float32).reshape(1,1,512,512)

    
    cv2.imshow("original", frame)
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(recv_data.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (512,512))
    print(np.unique(th, return_counts=True))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
c
