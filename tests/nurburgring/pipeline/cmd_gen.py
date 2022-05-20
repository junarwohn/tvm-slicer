import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()


partition_points = args.partition_points
last_index = partition_points[-1]
first_index = partition_points[0]

client_side = []
server_side = []

def combination(l):
    if len(l) == 1:
        return [l]
    else:
        combi = combination(l[1:])
        result = [[l[0]]] + [[l[0]] + i for i in combi] + combi
        return result

combi = combination(partition_points[1:-1])
# for com in combi:
#     # print("python3 client_revise.py -m {} -o {} -i {} -t {} -v 1 -p".format(args.model, args.opt_level, args.img_size, args.target), *([first_index] + com))
#     print("python3 client_revise.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), *([first_index] + com))
#     print("sleep 5")
# #     print("python3 client_revise.py -m {} -o {} -i {} -t {} -v 1 -p".format(args.model, args.opt_level, args.img_size, args.target), *([first_index] + com), ">> client_log.txt")
#     print("python3 client_revise.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), *([first_index] + com), ">> client_log.txt")
#     # print("python3 client_revise.py -m {} -o {} -i {} -t {} -p".format(args.model, args.opt_level, args.img_size, args.target), *([first_index] + com), ">> client_log.txt")
#     print("sleep 5")

for com in combi:
    print("python3 server_revise.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), com[-1], last_index)
    print("python3 server_revise.py -m {} -o {} -i {} -t {} --ip {} -p".format(args.model, args.opt_level, args.img_size, args.target, args.ip), com[-1], last_index, ">> client_log.txt")
