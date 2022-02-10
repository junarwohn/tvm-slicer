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
from tvm.relay.testing.darknet import __darknetffi__

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 


parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='yolov3', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
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

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess
    elif model == 'yolov3':
        def preprocess(img):
            return cv2.resize(img, (416, 416)) / 255
        return preprocess

preprocess = make_preprocess(args.model, args.img_size)

# color 설정

blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
white_color = (255, 255, 255)

# Font 종류

fonts = [cv2.FONT_HERSHEY_SIMPLEX,
cv2.FONT_HERSHEY_PLAIN,
cv2.FONT_HERSHEY_DUPLEX,
cv2.FONT_HERSHEY_COMPLEX,
cv2.FONT_HERSHEY_TRIPLEX,
cv2.FONT_HERSHEY_COMPLEX_SMALL,
cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
cv2.FONT_ITALIC]

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

MODEL_NAME = "yolov3"

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.

cfg_path = "../src/model/darknet/yolov3.cfg"
weights_path = "../src/model/darknet/yolov3.weights"
lib_path = "../src/model/darknet/libdarknet2.0.so"

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {"data": data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
target = "cuda"
dev = tvm.cuda()

shape = {"data": data.shape}
print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

model = graph_executor.GraphModule(lib["default"](dev))

# Video Load

img_size = args.img_size 
cap = cv2.VideoCapture("../src/data/walk.mp4")

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0
timer_visualize = 0


synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)

synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())

coco_name = "coco.names"
coco_url = "../src/" + "data/" + coco_name
font_name = "arial.ttf"
font_url = "../src/" + "data/" + font_name
#coco_path = download_testdata(coco_url, coco_name, module="data")
coco_path = coco_url
#font_path = download_testdata(font_url, font_name, module="data")
font_path = font_url

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    

timer_toal_start = time.time()
while (cap.isOpened()):
    timer_exclude_network_start = time.time()
    ret, frame = cap.read()
    try:
        img = preprocess(frame)
    except:
        break
    img = img.transpose(2, 0, 1)

    timer_inference_start = time.time()
    model.set_input("data", tvm.nd.array(img.astype("float32")))
    model.run()
    thresh = 0.5
    nms_thresh = 0.45
    tvm_out = []
    for i in range(3):
        layer_out = {}
        layer_out["type"] = "Yolo"
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = model.get_output(i * 4 + 3).numpy()
        layer_out["biases"] = model.get_output(i * 4 + 2).numpy()
        layer_out["mask"] = model.get_output(i * 4 + 1).numpy()
        out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
        layer_out["output"] = model.get_output(i * 4).numpy().reshape(out_shape)
        layer_out["classes"] = layer_attr[4]
        tvm_out.append(layer_out)
    timer_inference += time.time() - timer_inference_start

    timer_visualize_start = time.time()
    
    frame = frame.transpose(2, 0, 1) / 255

    _, im_h, im_w = frame.shape
    dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
        (416, 416), (im_w, im_h), thresh, 1, tvm_out
    )
    last_layer = net.layers[net.n - 1]
    tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
    tvm.relay.testing.yolo_detection.show_detections(frame, dets, thresh, names, last_layer.classes)
    tvm.relay.testing.yolo_detection.draw_detections(
        font_path, frame, dets, thresh, names, last_layer.classes
    )
    timer_visualize += time.time() - timer_visualize_start

    frame = frame.transpose(1, 2, 0)
    # cv2.imshow("img-{}".format("aa"), frame)
    cv2.imshow("img-{}".format("aa"), cv2.resize(frame, (960, 540)))
    # cv2.imshow("img-{}".format("aa"), cv2.resize(img, (1920,1080)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network

print("total time :", timer_total)
print("inference time :", timer_inference)
print("exclude network time :", timer_exclude_network)
print("network time :", timer_network)
print("timer_visualize :", timer_visualize)
cap.release()
