rm client_log.txt
rm server_log.txt
echo $(date) >> client_log.txt
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 91
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 91 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 91
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 91 >> client_log.txt
sleep 5
