rm server_log.txt

echo $(date) >> server_log.txt

# Sleep and Warm up
sleep 5
python3 model_tester.py -p 0 141
python3 middle_server.py -m unet -t cuda -o 3 -i 512 -p 13 115 --ip=192.168.0.184 >> server_log.txt

# Sleep and Warm up
sleep 5
python3 model_tester.py -p 0 141
python3 middle_server.py -m unet -t cuda -o 3 -i 512 -p 33 97 --ip=192.168.0.184 >> server_log.txt
