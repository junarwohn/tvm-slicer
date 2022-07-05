rm client_log.txt
rm server_log.txt
echo $(date) >> client_log.txt
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 21 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 21 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 21 31 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 21 31 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 31 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 10 31 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 21 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 21 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 21 31 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 21 31 59 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31 59
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31 59 >> client_log.txt
sleep 5
