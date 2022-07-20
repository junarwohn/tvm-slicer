rm client_log.txt

read ip

python3 client_pickle.py -i 4 --ip=$ip 
sleep 3
python3 client_pickle.py -i 4  --ip=$ip >> client_log.txt
sleep 3


python3 client_pickle.py -i 8 --ip=$ip 
sleep 3
python3 client_pickle.py -i 8  --ip=$ip >> client_log.txt
sleep 3


python3 client_pickle.py -i 16 --ip=$ip 
sleep 3
python3 client_pickle.py -i 16  --ip=$ip >> client_log.txt
sleep 3


python3 client_pickle.py -i 32 --ip=$ip 
sleep 3
python3 client_pickle.py -i 32  --ip=$ip >> client_log.txt
sleep 3

python3 client_pickle.py -i 64 --ip=$ip 
sleep 3
python3 client_pickle.py -i 64  --ip=$ip >> client_log.txt
sleep 3

python3 client_pickle.py -i 128 --ip=$ip 
sleep 3
python3 client_pickle.py -i 128  --ip=$ip >> client_log.txt
sleep 3

python3 client_pickle.py -i 256 --ip=$ip 
sleep 3
python3 client_pickle.py -i 256  --ip=$ip >> client_log.txt
sleep 3

python3 client_pickle.py -i 512 --ip=$ip 
sleep 3
python3 client_pickle.py -i 512  --ip=$ip >> client_log.txt
sleep 3

python3 client_pickle.py -i 1024 --ip=$ip 
sleep 3
python3 client_pickle.py -i 1024  --ip=$ip >> client_log.txt
sleep 3