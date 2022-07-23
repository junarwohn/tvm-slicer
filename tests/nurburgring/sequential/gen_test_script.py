from argparse import ArgumentParser
import os
import json
import re
import numpy as np
from sklearn import datasets

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--frequency', '-f' , type=int, default='300', help='type target file')
args = parser.parse_args()

clock = input()
server_cmd = []
server_cmd_format = "python3 server.py -p {} {}"
client_cmd = []
client_cmd_format = "python3 client.py -f {} {} -b {} {}"

while True:
    cmd = input()
    if cmd == '#' or cmd == "":
        break
    points, info = cmd.split("|")
    points = points.split(",")
    info = info.split("->")
    # local
    if len(info) == 1:
        client_cmd.append("python3 local.py -p {} {}".format(*points))
    # splitted into 2
    elif len(info) == 2:
        # server->client
        if info[0] == 'server' and info[1] == 'client':
            server_cmd.append(server_cmd_format.format(*points[0:2]))
            client_cmd.append(client_cmd_format.format(points[0], "", *points[1:]))
        # client->server
        elif info[0] == 'client' and info[1] == 'server':
            client_cmd.append(client_cmd_format.format(*points[:2], points[-1], ""))
            server_cmd.append(server_cmd_format.format(*points[1:3]))
    # splitted into 3
    elif len(info) == 3:
        client_cmd.append(client_cmd_format.format(*points[0:2], *points[2:4]))
        server_cmd.append(server_cmd_format.format(*points[1:3]))

print("##############")
print("sudo nvidia-smi -lgc {},{}".format(args.frequency, args.frequency))
print("rm client_log.txt")
for i in client_cmd:
    print(i)
    print(i , ">> client_log.txt")

print("##############")
print("rm server_log.txt")
for i in server_cmd:
    print(i)
    print("sleep 3")
    print(i, ">> server_log.txt")
    print("sleep 3")
