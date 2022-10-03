read pw

echo $pw | sudo -S nvidia-smi -lgc 300,300
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_300_128.txt

echo $pw | sudo -S nvidia-smi -lgc 705,705
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_705_128.txt

echo $pw | sudo -S nvidia-smi -lgc 1200,1200 
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_1200_128.txt

echo $pw | sudo -S nvidia-smi -lgc 1605,1605
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_1605_128.txt

echo $pw | sudo -S nvidia-smi -lgc 1965,1965
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_1965_128.txt
