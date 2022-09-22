server_ip=192.168.0.184

python3 client.py -c 0 0 0 0 -q 0 -f 0 -b 111 --ip=$server_ip
sleep 5
python3 client.py -c 0 0 0 0 -q 0 -f 0 -b 111 --ip=$server_ip
sleep 5

python3 client.py -c 0 0 0 0 -q 1 -f 0 -b 119 --ip=$server_ip
sleep 5
python3 client.py -c 0 0 0 0 -q 1 -f 0 -b 119 --ip=$server_ip
sleep 5

python3 client.py -c 0 0 0 0 -q 2 -f 0 -b 141 --ip=$server_ip
sleep 5
python3 client.py -c 0 0 0 0 -q 2 -f 0 -b 141 --ip=$server_ip
sleep 5

##
python3 client.py -c 1 0 0 0 -q 0 -f 0 -b 126 --ip=$server_ip
sleep 5
python3 client.py -c 1 0 0 0 -q 0 -f 0 -b 126 --ip=$server_ip
sleep 5

python3 client.py -c 1 0 0 0 -q 1 -f 0 -b 133 --ip=$server_ip
sleep 5
python3 client.py -c 1 0 0 0 -q 1 -f 0 -b 133 --ip=$server_ip
sleep 5

python3 client.py -c 1 0 0 0 -q 2 -f 0 -b 159 --ip=$server_ip
sleep 5
python3 client.py -c 1 0 0 0 -q 2 -f 0 -b 159 --ip=$server_ip
sleep 5
            
python3 client.py -c 2 0 0 0 -q 0 -f 0 -b 142 --ip=$server_ip
sleep 5
python3 client.py -c 2 0 0 0 -q 0 -f 0 -b 142 --ip=$server_ip
sleep 5

python3 client.py -c 2 0 0 0 -q 1 -f 0 -b 149 --ip=$server_ip
sleep 5
python3 client.py -c 2 0 0 0 -q 1 -f 0 -b 149 --ip=$server_ip
sleep 5

python3 client.py -c 2 0 0 0 -q 2 -f 0 -b 179 --ip=$server_ip
sleep 5
python3 client.py -c 2 0 0 0 -q 2 -f 0 -b 179 --ip=$server_ip
sleep 5
