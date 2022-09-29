BUILD=False
CLIENT=False
SERVER=False
VISUALIZE=False

# BUILD=True
# CLIENT=True
SERVER=True
# VISUALIZE=True

if CLIENT:
    print("server_ip=192.168.0.184")
if SERVER:
    print("server_ip=172.17.0.2")


cmd = ''
while cmd != '%':
    cmd = input()
    if len(cmd) == 0:
        continue
    if cmd[0] == '#':
        cmd = cmd[2:]
        c, q = cmd.split(', ')
        c = list(map(int, c[2:].split(' ')))
        q = int(q[2:])
    if cmd[0] == '0':
        p = cmd.split(' ')
        if BUILD:
            print("python3 slicing_graph.py -c {} {} {} {} -q {} -p {} {} {} {}".format(*c, q, *p))
        if CLIENT:
            print("python3 client.py -c {} {} {} {} -q {} -f {} {} -b {} {} --ip=$server_ip".format(*c, q, *p))
            print("sleep 3")
            print("python3 client.py -c {} {} {} {} -q {} -f {} {} -b {} {} --ip=$server_ip".format(*c, q, *p))
            print("sleep 3")
        if SERVER:
            print("python3 server.py -c {} {} {} {} -q {} -p {} {} --ip=$server_ip".format(*c, q, *p[1:3]))
            print("python3 server.py -c {} {} {} {} -q {} -p {} {} --ip=$server_ip".format(*c, q, *p[1:3]))
