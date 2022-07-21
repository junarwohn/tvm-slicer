echo "cass1234!" | sudo -S nvidia-smi -lgc 705,705
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_705.txt

echo "cass1234!" | sudo -S nvidia-smi -lgc 1200,1200 
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_1200.txt

echo "cass1234!" | sudo -S nvidia-smi -lgc 1605,1605
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_1605.txt

echo "cass1234!" | sudo -S nvidia-smi -lgc 2100,2100
bash ./model_test_script.sh
mv model_test_log.txt model_test_log_2080ti_2100.txt
