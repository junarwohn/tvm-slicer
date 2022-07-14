import numpy as np
from argparse import ArgumentParser
from itertools import combinations

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--front', '-f', nargs='+', type=int, default=0, help='set front point')
parser.add_argument('--back', '-b', nargs='+', type=int, default=0, help='set back point')
parser.add_argument('--name', '-n', type=str, default='client', help='set program name')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()


partition_points = args.partition_points
last_index = partition_points[-1]
first_index = partition_points[0]

client_side = []
server_side = []

print("rm client_log.txt")
print("rm server_log.txt")
combi = []
for i in range(len(partition_points) - 1):
    combi += list([list(c) for c in combinations(partition_points[1:-1], i)])

print("echo $(date) >> client_log.txt")
for com in combi:
    print("python3 client_pipeline_enabled.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), *([first_index] + com))
    print("sleep 5")
    print("python3 client_pipeline_enabled.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), *([first_index] + com), ">> client_log.txt")
    print("sleep 5")

print("echo $(date) >> server_log.txt")