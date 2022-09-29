    for j in 0 1 2
    do
        for q in 0 2
        do
            json_file=$(ls | grep UNet_M'\['$j'-0-0-0]_Q\['$q']_S\[0-' | grep json$)
            sp=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[0])")
            ep=$(echo $json_file | python3 -c "cmd=input(); p=cmd.split('_S[')[-1].split(']')[0].split('-'); print(list(map(int,p))[-1])")
            python3 visualize_mod.py -c $j 0 0 0 -q $q -p $sp $ep
        done
    done