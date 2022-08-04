rm client_log.txt
python3 local.py -p 0 141
sleep 3
python3 local.py -p 0 141 >> client_log.txt
sleep 3
python3 client.py -f 0 115 -b 141 
sleep 3
python3 client.py -f 0 115 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 133 -b 141 
sleep 3
python3 client.py -f 0 133 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 126 -b 141 
sleep 3
python3 client.py -f 0 126 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 127 -b 141 
sleep 3
python3 client.py -f 0 127 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0  -b 141 
sleep 3
python3 client.py -f 0  -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 109 -b 141 
sleep 3
python3 client.py -f 0 109 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 108 -b 141 
sleep 3
python3 client.py -f 0 108 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 97 -b 141 
sleep 3
python3 client.py -f 0 97 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0 11 -b 141 
sleep 3
python3 client.py -f 0 11 -b 141  >> client_log.txt
sleep 3
python3 client.py -f 0  -b 11 141
sleep 3
python3 client.py -f 0  -b 11 141 >> client_log.txt
sleep 3
python3 client.py -f 0  -b 5 141
sleep 3
python3 client.py -f 0  -b 5 141 >> client_log.txt
sleep 3
python3 client.py -f 0  -b 13 141
sleep 3
python3 client.py -f 0  -b 13 141 >> client_log.txt
sleep 3
python3 client.py -f 0  -b 25 141
sleep 3
python3 client.py -f 0  -b 25 141 >> client_log.txt
sleep 3
mv client_log.txt real_test_300.txt
