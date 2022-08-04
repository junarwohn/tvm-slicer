rm client_log.txt
python3 client.py -f 0  -b 141 
sleep 3
python3 client.py -f 0  -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0 11 -b 141 
sleep 3
python3 client.py -f 0 11 -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0 5 -b 141 
sleep 3
python3 client.py -f 0 5 -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0 13 -b 141 
sleep 3
python3 client.py -f 0 13 -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0  -b 133 141
sleep 3
python3 client.py -f 0  -b 133 141>> client_log.txt
sleep 3
python3 client.py -f 0 19 -b 141 
sleep 3
python3 client.py -f 0 19 -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0 25 -b 141 
sleep 3
python3 client.py -f 0 25 -b 141 >> client_log.txt
sleep 3
python3 client.py -f 0 27 -b 141 
sleep 3
python3 client.py -f 0 27 -b 141 >> client_log.txt
sleep 3
mv client_log.txt real_test_405.txt
