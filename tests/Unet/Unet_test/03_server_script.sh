#-c 0 0 0 0 -q 0
#############
#0 9 94 111
#//0 19 79 111
#++++++++++++
server_ip=""
python3 server.py -c 0 0 0 0 -q 0 -p 9 94 --ip=$server_ip

python3 server.py -c 0 0 0 0 -q 1 -p 9 107 --ip=$server_ip

python3 server.py -c 0 0 0 0 -q 2 -p 11 115 --ip=$server_ip
 
python3 server.py -c 1 0 0 0 -q 0 -p 20,66 66,126 --ip=$server_ip

python3 server.py -c 1 0 0 0 -q 1 -p 29,132 132,113 --ip=$server_ip

python3 server.py -c 1 0 0 0 -q 1 -p 25,171 144,171 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 0 -p 39,89 167,89 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 1 -p 39,155 155,146 --ip=$server_ip

python3 server.py -c 2 0 0 0 -q 2 -p 44,201 201,174 --ip=$server_ip