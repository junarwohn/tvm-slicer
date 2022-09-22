client = []
client_str = "python3 client.py -c {} -q {} -f {} {} -b {} {} --ip=$server_ip"
server = []
server_str = "python3 server.py -c {} -q {} -p {} {} --ip=$server_ip"

c = 0
q = 0
while True:
    a = input()
    if len(a) == 0:
        continue
    if a[0] == '@':
        break
    if a[0] == '#':
        if '-c' in a:
            info = a.split('-c ')[-1].split(' -q ')
            c = info[0]
            q = info[-1]
        continue
    slice_info = ['_'.join(i.split(',')) for i in a.split(' -p ')[-1].split(' ')]
    print(client_str.format(c, q, *slice_info[:2], *slice_info[2:4]))
    # print(server_str.format(c, q, *slice_info[1:3]))
