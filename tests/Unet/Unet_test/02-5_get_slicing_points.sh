for c in 0 1 2
do
    for q in 0 2
    do 
        python3 get_all_possibilities.py -c $c 0 0 0 -q $q
    done
done