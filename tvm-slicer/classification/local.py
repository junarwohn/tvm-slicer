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
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
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

model_path = "../src/model/{}_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

# Video Load

img_size = args.img_size 
cap = cv2.VideoCapture("../src/data/frames/output.mp4")

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0



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

timer_toal_start = time.time()
while (cap.isOpened()):
    timer_exclude_network_start = time.time()
    ret, frame = cap.read()
    try:
        frame = preprocess(frame)
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    timer_inference_start = time.time()
    model.set_input("input_1", input_data)
    model.run()
    outd = model.get_output(0)
    out = outd.numpy().astype(np.float32)
    #print(out.shape)
    top1_keras = np.argmax(out)
    out = synset[top1_keras]
    timer_inference += time.time() - timer_inference_start

    img_in_rgb = frame
    point = 30, 30 + 40
    img_in_rgb = cv2.resize(img_in_rgb, (512, 512))
    cv2.putText(img_in_rgb, out, point, fonts[0], 2, green_color, 2, cv2.LINE_AA)
    # cv2.imshow("received - client", img_in_rgb)
    # cv2.waitKey(1)
    # #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network

print("total time :", timer_total)
print("inference time :", timer_inference)
print("exclude network time :", timer_exclude_network)
print("network time :", timer_network)

#print("data receive size :", total_recv_msg_size)
#print("data send size :", total_send_msg_size)

cap.release()

import json
import pygraphviz as pgv

# os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

def shape_size(shape_list):
    result = 1
    for i in shape_list:
        result *= i
    return result

# def show_graph(json_data, file_name=None):
#     if type(json_data) == str:
#         json_data = json.loads(json_data)
#     A = pgv.AGraph(directed=True)
#     for node_idx, node in enumerate(json_data['nodes']):
#         for src in node['inputs']:
#             A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(shape_size(json_data['attrs']['shape'][1][src[0]])), node['name'] + '[{}]'.format(node_idx) + '{}'.format(shape_size(json_data['attrs']['shape'][1][node_idx])))
#     if file_name:
#         A.draw(file_name + '.png', format='png', prog='dot')

# show_graph(lib['get_graph_json'](), "resnet_{}_lv_{}".format(target, 3))

# with open("resnet_{}_lv_{}".format(target, 3), "w") as json_file:
#     json_file.write(lib['get_graph_json']())
