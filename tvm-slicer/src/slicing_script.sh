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

while getopts ":r:f:d:p:i:m:t:o:e:" opt; 
do
	case $opt in
		r) ratio="$OPTARG"
			;;
		f) file_name="$OPTARG"
			;;
		p) slice_point="$OPTARG"
			;;
		i) img_size="$OPTARG"
			;;
		m) model="$OPTARG"
			;;
		t) target="$OPTARG"
			;;
		o) opt_level="$OPTARG"
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
	cmd="python3 slicing_graph.py --start_point=0 --partition_point=${i} --end_point=${nodes[-1]} --img_size=${img_size} --model=${model} --target=${target} --opt_level=${opt_level}"
	echo $cmd
	$cmd
	cmd="python3 build_model.py --start_point=0 --partition_point=${i} --end_point=${nodes[-1]} --img_size=${img_size} --model=${model} --target=${target} --opt_level=${opt_level}"
	echo $cmd
	$cmd

done


