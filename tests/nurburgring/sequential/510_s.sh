read pw
echo $pw sudo -S nvidia-smi -lgc 510,510
rm server_log.txt
python3 server.py -p 0 141
python3 server.py -p 0 141>> server_log.txt
python3 server.py -p 11 141
python3 server.py -p 11 141>> server_log.txt
python3 server.py -p 5 141
python3 server.py -p 5 141>> server_log.txt
python3 server.py -p 13 141
python3 server.py -p 13 141>> server_log.txt
python3 server.py -p 0 133
python3 server.py -p 0 133>> server_log.txt
python3 server.py -p 19 141
python3 server.py -p 19 141>> server_log.txt
python3 server.py -p 25 141
python3 server.py -p 25 141>> server_log.txt
python3 server.py -p 27 141
python3 server.py -p 27 141>> server_log.txt
python3 server.py -p 11 133
python3 server.py -p 11 133>> server_log.txt
python3 server.py -p 33 141
python3 server.py -p 33 141>> server_log.txt
python3 server.py -p 5 133
python3 server.py -p 5 133>> server_log.txt
