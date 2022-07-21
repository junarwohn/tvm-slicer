from argparse import ArgumentParser
import os
import json
import re

from matplotlib.pyplot import hist

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--file', '-f' , type=str, default='model_test_log.txt', help='type target file')
args = parser.parse_args()

current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"


# lookuptable : { str(src) : [str(dst1), str(dst2) ...] }
lut_src_dst = dict()
# lookuptable : { str(src-dst) : time }
lut_path_time = dict()

f = open("./" + args.file, 'r')
# path_pat = "\[\d+, \d+\]"
# data format
while True:
    line = f.readline()
    line = line.strip()
    if not line:
        break
    src, dst = [int(i) for i in re.findall(r"\d+", re.findall(r"\[\d+, \d+\]", line)[0])]
    time = float(re.findall(r"\d+\.\d+", line)[0])
    if src in lut_src_dst:
        lut_src_dst[src].append(dst)
    else:
        lut_src_dst[src] = [dst]
    
    lut_path_time["{}_{}".format(src, dst)] = time

# sort lut
for key in lut_src_dst:
    lut_src_dst[key] = sorted(lut_src_dst[key])


# check total time
#init
start_node = 0
end_node = lut_src_dst[0][-1]
def dfs(history, end_node, lut, cost_lut):
    last_visit = history[-1]
    if last_visit == end_node:
        print(history, sum([cost_lut["{}_{}".format(history[i], history[i+1])] for i in range(len(history) - 1)]))
    else:
        for dst in lut[last_visit]:
            dfs(history+[dst], end_node, lut, cost_lut)

dfs([start_node], end_node, lut_src_dst, lut_path_time)