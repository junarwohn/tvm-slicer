#python3 server.py  --target=cuda --ip=127.0.0.1 --ntp_enable=0 --partition_point=20 --model=unet --opt_level=3

bash client.sh -r 40 -f graph_candidate.txt -i 512 -m unet -t cuda -o 3 -a 192.168.0.184 -n 0
