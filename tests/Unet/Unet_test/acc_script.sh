time_info='220923-0907'
dir_name="model_"$time_info

for i in 1 2 3 4 5
do
    # echo './'$dir_name"_"$i'/'
    cd './'$dir_name"_"$i'/'
    rm acc_result.txt
    for j in 0 1 2
    do
        for q in 0 2
        do
            python3 ../slicing_graph.py -c $j 0 0 0 -b 1 -q $q
            echo c: $j 0 0 0 q: $q >> acc_result.txt
            json_file=$(ls | grep UNet_M'\['$j'-0-0-0]_Q\['$q']_S\[0-')
            sp=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[0])")
            ep=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[-1])")
            # python3 ../acc_test.py -c $j 0 0 0 -q $q -p $sp $ep >> acc_result.txt
        done
    done
    cd ..
done
