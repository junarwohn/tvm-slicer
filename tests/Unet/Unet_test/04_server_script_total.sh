#-c 0 0 0 0 -q 0
#############
#0 9 94 111
#//0 19 79 111
#++++++++++++
server_ip=172.17.0.2
python3 server.py -c 0 0 0 0 -q 0 -p 0 111 --ip=$server_ip
python3 server.py -c 0 0 0 0 -q 0 -p 0 111 --ip=$server_ip

python3 server.py -c 0 0 0 0 -q 1 -p 0 119 --ip=$server_ip
python3 server.py -c 0 0 0 0 -q 1 -p 0 119 --ip=$server_ip

python3 server.py -c 0 0 0 0 -q 2 -p 0 141 --ip=$server_ip
python3 server.py -c 0 0 0 0 -q 2 -p 0 141 --ip=$server_ip
 
python3 server.py -c 1 0 0 0 -q 0 -p 0 126 --ip=$server_ip
python3 server.py -c 1 0 0 0 -q 0 -p 0 126 --ip=$server_ip

python3 server.py -c 1 0 0 0 -q 1 -p 0 133 --ip=$server_ip
python3 server.py -c 1 0 0 0 -q 1 -p 0 133 --ip=$server_ip

python3 server.py -c 1 0 0 0 -q 2 -p 0 159 --ip=$server_ip
python3 server.py -c 1 0 0 0 -q 2 -p 0 159 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 0 -p 0 142 --ip=$server_ip
python3 server.py -c 2 0 0 0 -q 0 -p 0 142 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 1 -p 0 149 --ip=$server_ip
python3 server.py -c 2 0 0 0 -q 1 -p 0 149 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 2 -p 0 179 --ip=$server_ip
python3 server.py -c 2 0 0 0 -q 2 -p 0 179 --ip=$server_ip
