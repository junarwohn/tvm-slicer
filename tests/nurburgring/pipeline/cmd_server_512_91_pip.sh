rm client_log.txt
rm server_log.txt
echo $(date) >> server_log.txt
python3 server_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 91 119
python3 server_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 91 119 >> server_log.txt
