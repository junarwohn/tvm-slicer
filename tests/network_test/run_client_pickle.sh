rm client_log.txt

python3 client_pickle.py -i 4
sleep 3
python3 client_pickle.py -i 4 >> client_log.txt
sleep 3


python3 client_pickle.py -i 8
sleep 3
python3 client_pickle.py -i 8 >> client_log.txt
sleep 3


python3 client_pickle.py -i 16
sleep 3
python3 client_pickle.py -i 16 >> client_log.txt
sleep 3


python3 client_pickle.py -i 32
sleep 3
python3 client_pickle.py -i 32 >> client_log.txt
sleep 3

python3 client_pickle.py -i 64
sleep 3
python3 client_pickle.py -i 64 >> client_log.txt
sleep 3

python3 client_pickle.py -i 128
sleep 3
python3 client_pickle.py -i 128 >> client_log.txt
sleep 3

python3 client_pickle.py -i 256
sleep 3
python3 client_pickle.py -i 256 >> client_log.txt
sleep 3

python3 client_pickle.py -i 512
sleep 3
python3 client_pickle.py -i 512 >> client_log.txt
sleep 3

python3 client_pickle.py -i 1024
sleep 3
python3 client_pickle.py -i 1024 >> client_log.txt
sleep 3