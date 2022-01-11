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

#target = 'cuda'
target = 'llvm'
#dev = tvm.cuda(0)
dev = tvm.cpu(0)

model_path = "../src/model/unet_tvm.so"
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

# Video Load

img_size = 128
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
stime = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (img_size, img_size)) / 255
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    # Execute front
    model.set_input("input_1", input_data)
    model.run()

    out = model.get_output(0).asnumpy().astype(np.float32)

    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    print(np.unique(th, return_counts=True))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
print("Total time :", time.time() - stime)
cap.release()
