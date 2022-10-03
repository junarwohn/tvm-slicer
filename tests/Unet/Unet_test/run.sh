for i in 1 2 3 ... 10
do
	python3 build_graph.py -p 0 165 -c 2 0 0 0 -q 2 -b 1
	python3 visualize_mod.py -p 0 165 -c 2 0 0 0 -q 2
done
