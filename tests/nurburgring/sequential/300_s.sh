read pw
echo $pw | sudo -S nvidia-smi -lgc 300,300
rm server_log.txt
python3 server.py -p  55  61

python3 server.py -p  55  61 >> server_log.txt

python3 server.py -p  41  53

python3 server.py -p  41  53 >> server_log.txt

python3 server.py -p  53  55

python3 server.py -p  53  55 >> server_log.txt

python3 server.py -p  41  47

python3 server.py -p  41  47 >> server_log.txt

python3 server.py -p  41  79

python3 server.py -p  41  79 >> server_log.txt

python3 server.py -p  53  61

python3 server.py -p  53  61 >> server_log.txt

python3 server.py -p  41  55

python3 server.py -p  41  55 >> server_log.txt

python3 server.py -p  47  53

python3 server.py -p  47  53 >> server_log.txt

python3 server.py -p  41  61

python3 server.py -p  41  61 >> server_log.txt

python3 server.py -p  55  72

python3 server.py -p  55  72 >> server_log.txt
