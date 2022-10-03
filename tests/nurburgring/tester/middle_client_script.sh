rm client_log.txt

echo $(date) >> client_log.txt

# Sleep and Warm up
sleep 5
python3 model_tester.py -p 0 141
python3 middle_client.py -m unet -t cuda -o 3 -i 512 -f 0 13 -b 115 141 --ip=192.168.0.184 >> client_log.txt

# Sleep and Warm up
sleep 5
python3 model_tester.py -p 0 141
python3 middle_client.py -m unet -t cuda -o 3 -i 512 -f 0 33 -b 97 141 --ip=192.168.0.184 >> client_log.txt
