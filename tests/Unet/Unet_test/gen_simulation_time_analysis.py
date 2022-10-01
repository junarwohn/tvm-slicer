import os
from itertools import permutations

BASE_DIR='slice_config'
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
    if cmd[0] == '$':
        
        cmd = cmd[2:]
        p = list(map(int, cmd.split(' ')))
        if p[0] != 0:
            p = [0] + p
        s = p[0]
        e = p[-1]
        candidates = list(permutations(p[1:-1], 2))
        for ms, me in candidates:
            if me - ms < (e - s) / 2:
                continue
            print("python3 simulator.py -c {} {} {} {} -b 0 -q {} -p {} {} {} {}".format(*c, q, s, ms, me, e))