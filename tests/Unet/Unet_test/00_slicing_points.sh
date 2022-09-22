# ******************
#-c 0 0 0 0 -q 0
# 9 19 29 39 55 70 85 100 111
#############
python3 slicing_graph.py -c 0 0 0 0 -q 0 -p 0 9 100 111
python3 slicing_graph.py -c 0 0 0 0 -q 0 -p 0 19 85 111
python3 slicing_graph.py -c 0 0 0 0 -q 0 -p 0 39 55 111

#-c 0 0 0 0 -q 2
# 5 11 13 19 25 27 33 39 41 47 53 55 61 72 73 79 90 91 97 108 109 115 126 127 133 141
############
python3 slicing_graph.py -c 0 0 0 0 -q 2 -p 0 11 133 141
python3 slicing_graph.py -c 0 0 0 0 -q 2 -p 0 13 115 141
python3 slicing_graph.py -c 0 0 0 0 -q 2 -p 0 41 73 141


# ******************
#-c 1 0 0 0 -q 0
# 29 39 49 54 70 85 100 115 126
#############
python3 slicing_graph.py -c 1 0 0 0 -q 0 -p 0 29,54 100,54 126
python3 slicing_graph.py -c 1 0 0 0 -q 0 -p 0 39,54 85,54 126

#-c 1 0 0 0 -q 2
# 5 12 18 20 26 32 38 40 46 52 54 60 66 68 74 85 86 92 103 104 110 121 122 128 139 144 145 151 159
#############
python3 slicing_graph.py -c 1 0 0 0 -q 2 -p 0 18 151 159
python3 slicing_graph.py -c 1 0 0 0 -q 2 -p 0 20,144 128,144 159


# ******************
#-c 2 0 0 0 -q 0
# 40 50 60 70 86 101 116 131 142
#############
python3 slicing_graph.py -c 2 0 0 0 -q 0 -p 0 40,70 116,70 142
python3 slicing_graph.py -c 2 0 0 0 -q 0 -p 0 50,70 101,70 142

#-c 2 0 0 0 -q 2
# 5 12 18 20 26 28 34 40 46 52 54 60 66 68 74 80 82 88 99 100 106 117 118 124 135 136 142 153 158 164 165 171 179
#############
python3 slicing_graph.py -c 2 0 0 0 -q 2 -p 0 18 165 179
python3 slicing_graph.py -c 2 0 0 0 -q 2 -p 0 28,158 142,158 179
# @