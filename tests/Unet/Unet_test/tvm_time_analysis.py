import numpy as np
import tvm
from tvm.contrib import graph_executor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
args = parser.parse_args()

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
model_config = args.model_config
quantization_level = args.quantization_level
model_path = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so".format(*model_config)
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

total_frames = 100
indata = tvm.nd.array(np.random.normal(0,1,(1,3,img_size, img_size)).astype('float32'), device=dev)

print(
    'set_input (ms) :', 
    1000 * model.module.time_evaluator(func_name='set_input', dev=dev, number=total_frames)('input_1', indata).results[0])
print(
    'run (ms) :', 
    1000 * model.module.time_evaluator(func_name='run', dev=dev, number=total_frames)().results[0])
