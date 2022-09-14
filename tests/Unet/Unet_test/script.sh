#!/bin/bash
for i in {5..20}
do
    # rm -r model_$i
    mkdir model_$i
    python3 Unet_train.py
    mv *h5 model_$i/
done