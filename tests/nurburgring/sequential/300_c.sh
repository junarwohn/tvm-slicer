rm client_log.txt
python3 client.py -f 0  55 -b  61  141
sleep 3
python3 client.py -f 0  55 -b  61  141 >> client_log.txt
sleep 3
python3 client.py -f 0  41 -b  53  141
sleep 3
python3 client.py -f 0  41 -b  53  141 >> client_log.txt
sleep 3
python3 client.py -f 0  53 -b  55  141
sleep 3
python3 client.py -f 0  53 -b  55  141 >> client_log.txt
sleep 3
python3 client.py -f 0  41 -b  47  141
sleep 3
python3 client.py -f 0  41 -b  47  141 >> client_log.txt
sleep 3
python3 client.py -f 0  41 -b  79  141
sleep 3
python3 client.py -f 0  41 -b  79  141 >> client_log.txt
sleep 3
python3 client.py -f 0  53 -b  61  141
sleep 3
python3 client.py -f 0  53 -b  61  141 >> client_log.txt
sleep 3
python3 client.py -f 0  41 -b  55  141
sleep 3
python3 client.py -f 0  41 -b  55  141 >> client_log.txt
sleep 3
python3 client.py -f 0  47 -b  53  141
sleep 3
python3 client.py -f 0  47 -b  53  141 >> client_log.txt
sleep 3
python3 client.py -f 0  41 -b  61  141
sleep 3
python3 client.py -f 0  41 -b  61  141 >> client_log.txt
sleep 3
python3 client.py -f 0  55 -b  72  141
sleep 3
python3 client.py -f 0  55 -b  72  141 >> client_log.txt
sleep 3
mv client_log.txt real_test_300.txt