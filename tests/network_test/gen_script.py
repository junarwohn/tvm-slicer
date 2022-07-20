from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--name', '-n', type=str, default='client', help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

if args.name == "client":
    program_name = 'client_pickle.py'
    print("rm client_log.txt")
else:
    program_name = 'server_pickle.py'
    print("rm server_log.txt")

print("read ip")

for i in range(100):
    if args.name == 'client':
        print("python3 client_pickle.py -i", 100*(i+1), "--ip=$ip")
        print("sleep 3")
        print("python3 client_pickle.py -i", 100*(i+1), "--ip=$ip >> client_log.txt")
        print("sleep 3")
    else:
        print("python3 server_pickle.py -i", 100*(i+1), "--ip=$ip")
        print("python3 server_pickle.py -i", 100*(i+1), "--ip=$ip")