time_info=`date +%y%m%d-%H%M`
dir_name="model_"$time_info
echo $dir_name
# python3 Unet_train.py > $dir_name"_unet_train_log.txt"

for i in 1 2 3 4 5
do
    echo $dir_name"_"$i
    mkdir $dir_name"_"$i
    python3 Unet_train.py > "unet_train_log.txt"
    mv "unet_train_log.txt" $dir_name"_"$i
    mv *h5 './'$dir_name"_"$i'/'
done
