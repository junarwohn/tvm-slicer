read pw
echo $pw sudo -S nvidia-smi -lgc 300,300
rm server_log.txt
python3 server.py -p 115 141
python3 server.py -p 115 141 >> server_log.txt
python3 server.py -p 133 141
python3 server.py -p 133 141 >> server_log.txt
python3 server.py -p 127 141
python3 server.py -p 127 141 >> server_log.txt
python3 server.py -p 126 141
python3 server.py -p 126 141 >> server_log.txt
python3 server.py -p 0 141
python3 server.py -p 0 141 >> server_log.txt
python3 server.py -p 109 141
python3 server.py -p 109 141 >> server_log.txt
python3 server.py -p 108 141
python3 server.py -p 108 141 >> server_log.txt
python3 server.py -p 97 141
python3 server.py -p 97 141 >> server_log.txt
python3 server.py -p 11 141
python3 server.py -p 11 141 >> server_log.txt
python3 server.py -p 0 11
python3 server.py -p 0 11 >> server_log.txt
python3 server.py -p 0 5
python3 server.py -p 0 5 >> server_log.txt
python3 server.py -p 0 13
python3 server.py -p 0 13 >> server_log.txt
python3 server.py -p 0 25
python3 server.py -p 0 25 >> server_log.txt
