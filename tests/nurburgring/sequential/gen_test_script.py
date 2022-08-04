from argparse import ArgumentParser
import os
import json
import re
import numpy as np
from sklearn import datasets

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--frequency', '-f' , type=int, default='300', help='type target file')
parser.add_argument('--print_client', '-c', type=int, default=1)
args = parser.parse_args()

server_cmd = []
server_cmd_format = "python3 server.py -p {} {}"
client_cmd = []
client_cmd_format = "python3 client.py -f {} {} -b {} {}"

while True:
    try:
        cmd = input()
    except:
        break
    if cmd == '#' or cmd == "" or cmd =="\n":
        break
    points, info = cmd.split("|")
    points = points.split(",")
    info = info.split(",")
    # local or cloud
    if len(info) == 1:
        if info[0] == 'server':
            client_cmd.append(client_cmd_format.format(points[0], '', points[1], ''))
            server_cmd.append(server_cmd_format.format(*points[0:2]))
        else:
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

f_c = open("./{}_c.sh".format(args.frequency), 'w')
f_s = open("./{}_s.sh".format(args.frequency), 'w')

# if args.print_client == 1:
f_c.write("rm client_log.txt")
f_c.write('\n')
for i in client_cmd:
    f_c.write(i)
    f_c.write('\n')
    f_c.write("sleep 3")
    f_c.write('\n')
    f_c.write(str(i) + ">> client_log.txt")
    f_c.write('\n')
    f_c.write("sleep 3")
    f_c.write('\n')
f_c.write("mv client_log.txt real_test_{}.txt".format(args.frequency))
f_c.write('\n')
# else:
f_s.write("read pw")
f_s.write('\n')
f_s.write("echo $pw sudo -S nvidia-smi -lgc {},{}".format(args.frequency, args.frequency))
f_s.write('\n')
f_s.write("rm server_log.txt")
f_s.write('\n')
for i in server_cmd:
    f_s.write(i)
    f_s.write('\n')
    f_s.write(str(i) + ">> server_log.txt")
    f_s.write('\n')

f_c.close()
f_s.close()