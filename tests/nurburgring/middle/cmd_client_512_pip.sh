rm client_log.txt
rm server_log.txt
echo $(date) >> client_log.txt
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 31
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 31 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 31 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 31 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 20 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 31
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 31 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 31 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 31 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 9 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 31
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 31 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 31 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 31 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 20 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 31 42 >> client_log.txt
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 42
sleep 5
python3 client_pipeline_enabled.py -m unet -o 3 -i 512 -t cuda --ip 192.168.0.184 -p 0 42 >> client_log.txt
sleep 5
