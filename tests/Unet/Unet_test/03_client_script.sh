server_ip=""

python3 client.py -c 0 0 0 0 -q 0 -f 0 9 -b 94 111 --ip=$server_ip -v 1

python3 client.py -c 0 0 0 0 -q 1 -f 0 9 -b 107 119 --ip=$server_ip

python3 client.py -c 0 0 0 0 -q 2 -f 0 11 -b 115 141 --ip=$server_ip
            
python3 client.py -c 1 0 0 0 -q 0 -f 0 20,66 -b 66,126 148 --ip=$server_ip

python3 client.py -c 1 0 0 0 -q 1 -f 0 29,132 -b 132,113 155 --ip=$server_ip

python3 client.py -c 1 0 0 0 -q 2 -f 0 25,171 -b 144,171 186 --ip=$server_ip
            
python3 client.py -c 2 0 0 0 -q 0 -f 0 39,89 -b 167,89 184 --ip=$server_ip

python3 client.py -c 2 0 0 0 -q 1 -f 0 39,155 -b 155,146 191 --ip=$server_ip

python3 client.py -c 2 0 0 0 -q 2 -f 0 44,201 -b 201,174 233 --ip=$server_ip