rm -r slice_config
mkdir slice_config
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-65-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 65 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-65-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-75-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 75 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-75-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-76-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 76 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-76-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-81-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 81 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-81-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-5-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 5 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-5-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-75-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 75 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-75-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-76-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 76 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-76-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-81-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 81 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-81-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-10-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 10 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-10-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-75-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 75 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-75-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-76-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 76 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-76-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-81-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 81 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-81-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-11-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 11 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-11-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-76-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 76 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-76-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-81-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 81 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-81-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-16-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 16 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-16-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-81-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 81 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-81-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-21-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 21 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-21-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-22-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 22 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-22-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-91-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 91 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-91-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-27-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 27 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-27-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-32-92-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 32 92 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-32-92-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-32-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 32 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-32-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-32-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 32 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-32-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-32-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 32 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-32-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-32-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 32 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-32-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-33-97-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 33 97 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-33-97-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-33-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 33 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-33-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-33-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 33 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-33-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-33-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 33 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-33-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-38-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 38 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-38-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-38-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 38 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-38-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-38-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 38 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-38-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-43-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 43 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-43-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-43-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 43 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-43-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-43-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 43 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-43-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-44-107-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 44 107 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-44-107-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-44-108-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 44 108 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-44-108-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-44-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 44 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-44-113-120]
mkdir slice_config/M[0-0-0-0]_Q[0]_S[0-49-113-120]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 0 -p 0 49 113 120
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-120\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[0]_S[0-49-113-120]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-79-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 79 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-79-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-90-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 90 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-90-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-91-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 91 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-91-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-97-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 97 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-97-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-5-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 5 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-5-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-90-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 90 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-90-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-91-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 91 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-91-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-97-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 97 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-97-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-11-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 11 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-11-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-90-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 90 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-90-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-91-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 91 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-91-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-97-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 97 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-97-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-13-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 13 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-13-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-90-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 90 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-90-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-91-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 91 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-91-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-97-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 97 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-97-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-19-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 19 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-19-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-97-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 97 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-97-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-25-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 25 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-25-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-27-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 27 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-27-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-108-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 108 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-108-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-109-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 109 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-109-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-33-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 33 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-33-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-39-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 39 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-39-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-39-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 39 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-39-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-39-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 39 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-39-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-39-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 39 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-39-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-41-115-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 41 115 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-41-115-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-41-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 41 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-41-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-41-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 41 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-41-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-41-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 41 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-41-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-47-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 47 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-47-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-47-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 47 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-47-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-47-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 47 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-47-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-53-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 53 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-53-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-53-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 53 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-53-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-53-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 53 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-53-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-55-126-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 55 126 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-55-126-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-55-127-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 55 127 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-55-127-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-55-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 55 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-55-133-141]
mkdir slice_config/M[0-0-0-0]_Q[2]_S[0-61-133-141]
python3 slicing_graph.py -c 0 0 0 0 -b 0 -q 2 -p 0 61 133 141
ls | grep json$ | grep UNet | grep 'M\[0-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[0-0-0-0]_Q[2]_S[0-61-133-141]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-75-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 75 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-75-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-76-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 76 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-76-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-81-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 81 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-81-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-91-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 91 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-91-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-5-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 5 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-5-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-76-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 76 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-76-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-81-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 81 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-81-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-91-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 91 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-91-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-11-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 11 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-11-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-81-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 81 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-81-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-91-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 91 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-91-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-16-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 16 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-16-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-91-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 91 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-91-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-21-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 21 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-21-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-91-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 91 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-91-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-22-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 22 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-22-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-92-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 92 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-92-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-27-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 27 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-27-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-97-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 97 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-97-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-32-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 32 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-32-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-33-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 33 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-33-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-33-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 33 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-33-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-33-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 33 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-33-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-33-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 33 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-33-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-33-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 33 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-33-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-38-107-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 38 107 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-38-107-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-38-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 38 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-38-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-38-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 38 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-38-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-38-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 38 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-38-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-38-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 38 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-38-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-43-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 43 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-43-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-43-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 43 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-43-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-43-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 43 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-43-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-43-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 43 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-43-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-44-112-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 44 112 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-44-112-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-44-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 44 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-44-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-44-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 44 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-44-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-44-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 44 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-44-123-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-49-117-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 49 117 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-49-117-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-49-118-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 49 118 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-49-118-130]
mkdir slice_config/M[1-0-0-0]_Q[0]_S[0-49-123-130]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 0 -p 0 49 123 130
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-130\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[0]_S[0-49-123-130]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-89-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 89 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-89-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-90-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 90 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-90-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-96-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 96 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-96-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-107-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 107 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-107-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-5-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 5 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-5-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-89-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 89 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-89-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-90-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 90 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-90-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-96-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 96 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-96-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-107-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 107 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-107-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-12-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 12 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-12-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-96-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 96 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-96-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-107-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 107 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-107-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-18-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 18 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-18-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-107-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 107 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-107-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-24-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 24 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-24-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-107-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 107 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-107-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-26-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 26 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-26-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-108-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 108 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-108-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-32-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 32 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-32-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-114-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 114 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-114-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-38-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 38 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-38-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-40-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 40 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-40-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-40-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 40 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-40-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-40-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 40 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-40-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-40-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 40 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-40-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-40-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 40 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-40-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-46-125-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 46 125 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-46-125-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-46-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 46 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-46-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-46-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 46 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-46-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-46-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 46 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-46-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-46-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 46 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-46-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-52-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 52 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-52-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-52-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 52 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-52-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-52-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 52 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-52-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-52-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 52 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-52-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-54-130-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 54 130 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-54-130-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-54-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 54 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-54-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-54-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 54 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-54-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-54-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 54 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-54-143-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-60-136-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 60 136 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-60-136-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-60-137-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 60 137 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-60-137-151]
mkdir slice_config/M[1-0-0-0]_Q[2]_S[0-60-143-151]
python3 slicing_graph.py -c 1 0 0 0 -b 0 -q 2 -p 0 60 143 151
ls | grep json$ | grep UNet | grep 'M\[1-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-151\]' | xargs mv -t slice_config/M[1-0-0-0]_Q[2]_S[0-60-143-151]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-76-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 76 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-76-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-81-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 81 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-81-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-91-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 91 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-91-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-92-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 92 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-92-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-97-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 97 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-97-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-5-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 5 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-5-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-91-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 91 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-91-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-92-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 92 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-92-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-97-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 97 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-97-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-11-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 11 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-11-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-91-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 91 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-91-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-92-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 92 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-92-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-97-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 97 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-97-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-16-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 16 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-16-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-92-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 92 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-92-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-97-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 97 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-97-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-21-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 21 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-21-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-97-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 97 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-97-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-22-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 22 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-22-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-27-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 27 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-27-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-32-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 32 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-32-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-107-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 107 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-107-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-33-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 33 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-33-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-112-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 112 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-112-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-113-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 113 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-113-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-38-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 38 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-38-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-43-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 43 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-43-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-43-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 43 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-43-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-43-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 43 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-43-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-43-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 43 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-43-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-43-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 43 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-43-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-44-118-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 44 118 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-44-118-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-44-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 44 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-44-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-44-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 44 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-44-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-44-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 44 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-44-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-44-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 44 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-44-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-49-123-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 49 123 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-49-123-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-49-128-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 49 128 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-49-128-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-49-129-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 49 129 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-49-129-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-49-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 49 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-49-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-59-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 59 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-59-134-141]
mkdir slice_config/M[2-0-0-0]_Q[0]_S[0-60-134-141]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 0 -p 0 60 134 141
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[0\]' | grep -v full | grep -v 'S\[0-141\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[0]_S[0-60-134-141]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-89-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 89 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-89-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-90-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 90 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-90-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-96-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 96 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-96-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-107-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 107 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-107-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-108-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 108 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-108-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-114-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 114 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-114-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-5-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 5 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-5-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-96-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 96 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-96-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-107-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 107 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-107-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-108-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 108 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-108-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-114-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 114 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-114-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-12-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 12 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-12-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-107-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 107 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-107-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-108-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 108 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-108-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-114-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 114 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-114-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-18-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 18 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-18-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-107-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 107 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-107-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-108-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 108 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-108-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-114-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 114 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-114-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-24-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 24 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-24-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-114-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 114 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-114-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-26-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 26 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-26-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-32-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 32 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-32-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-38-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 38 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-38-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-125-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 125 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-125-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-40-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 40 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-40-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-130-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 130 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-130-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-132-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 132 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-132-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-46-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 46 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-46-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-52-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 52 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-52-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-52-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 52 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-52-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-52-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 52 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-52-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-52-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 52 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-52-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-52-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 52 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-52-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-54-138-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 54 138 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-54-138-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-54-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 54 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-54-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-54-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 54 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-54-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-54-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 54 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-54-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-54-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 54 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-54-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-60-144-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 60 144 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-60-144-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-60-150-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 60 150 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-60-150-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-60-151-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 60 151 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-60-151-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-60-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 60 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-60-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-71-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 71 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-71-157-165]
mkdir slice_config/M[2-0-0-0]_Q[2]_S[0-72-157-165]
python3 slicing_graph.py -c 2 0 0 0 -b 0 -q 2 -p 0 72 157 165
ls | grep json$ | grep UNet | grep 'M\[2-0-0-0\]_Q\[2\]' | grep -v full | grep -v 'S\[0-165\]' | xargs mv -t slice_config/M[2-0-0-0]_Q[2]_S[0-72-157-165]
