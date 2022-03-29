#python3 server.py  --target=cuda --ip=127.0.0.1 --ntp_enable=0 --partition_point=20 --model=unet --opt_level=3

bash autorun_server.sh -r 99 -f graph_candidate.txt -i 224 -m resnet152 -t cuda -o 3 -a 192.168.0.184 -n 0
