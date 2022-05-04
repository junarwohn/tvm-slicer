#!/bin/bash

#python3 slicing_graph.py --device=cuda --partition_point

# -r ratio of slice upper bound 
# -f file to read

# -d device
# -p slice point
# -e end point
# -i image size
# -m model
# -t target
# -o opt_level
# -v visualize

while getopts ":r:f:i:m:t:o:a:n:v:" opt; 
do
	case $opt in
		r) ratio="$OPTARG"
			;;
		f) file_name="$OPTARG"
			;;
		i) img_size="$OPTARG"
			;;
		m) model="$OPTARG"
			;;
		t) target="$OPTARG"
			;;
		o) opt_level="$OPTARG"
			;;
		a) ip_address="$OPTARG"
			;;
		n) ntp_enable="$OPTARG"
			;;
		v) visualize="$OPTARG"
			;;
		\?) echo "Invalid option -$OPTARG" >&2
			exit 1
			;;
	esac

	case $OPTARG in
		-*) echo "Option $opt needs a valid argument"
			exit 1
			;;
	esac
done

node_candidate=`cat ${file_name}`
nodes=($node_candidate)
len_nodes=${#nodes[@]}

#echo ${nodes[@]}
#for i in ${nodes[@]}
#do
#	echo $i
#done

upper_bound=$((len_nodes * ratio / 100))

for i in ${nodes[@]:0:$upper_bound}
#for i in ${nodes[@]:0:1}
do
	cmd="python3 client.py --ip=${ip_address} --partition_point=${i} --img_size=${img_size} --model=${model} --target=${target} --opt_level=${opt_level} --ntp_enable=${ntp_enable} --visualize=${visualize}"
	echo $cmd
	echo "server ${model} ${target} ${i} ${opt_level}" >> server_log.txt
 	$cmd >> server_log.txt
	sleep 3
done


