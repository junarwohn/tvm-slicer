import os
from itertools import permutations

BASE_DIR='slice_config'
cmd = ''
print("rm -r", BASE_DIR)
print("mkdir", BASE_DIR)
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
            target_dir="M[{}-{}-{}-{}]_Q[{}]_S[{}-{}-{}-{}]".format(*c, q, s, ms, me, e)
            dir_path="/".join([BASE_DIR, target_dir])
            print("mkdir", dir_path)
            print("python3 slicing_graph.py -c {} {} {} {} -b 0 -q {} -p {} {} {} {}".format(*c, q, s, ms, me, e))
            
            print("ls | grep json$ | grep UNet | grep 'M\[{}-{}-{}-{}\]_Q\[{}\]' | grep -v full | grep -v 'S\[{}-{}\]' | xargs mv -t {}".format(*c, q, s, e, dir_path))
