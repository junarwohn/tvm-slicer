#python3 server.py  --target=cuda --ip=127.0.0.1 --ntp_enable=0 --partition_point=20 --model=unet --opt_level=3

bash autorun_client.sh -r 99 -f unet_slice_points.txt -i 512 -m unet -t cuda -o 3 -a 192.168.0.184 -n 0 -v 1
