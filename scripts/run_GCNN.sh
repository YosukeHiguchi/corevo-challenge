#!/usr/bin/bash

stage=1
model_name=test
gpu=-1

# Network
epoch=50
batchsize=128
optimizer='Adam'

# data
feature='mfcc'

. utils/parse_options.sh

if [ $stage -le 1 ]; then
    python -W ignore run_GCNN.py --train \
            --gpu $gpu \
            --epoch $epoch \
            --batchsize $batchsize \
            --optimizer $optimizer \
            --feature $feature \
            --out GCNN/model/$model_name
fi

if [ $stage -le 2 ]; then
    python -W ignore run_GCNN.py --predict \
            --gpu $gpu \
            --model_path GCNN/model/$model_name/${epoch}.model \
            --feature $feature \
            --out GCNN/result/$model_name
fi
