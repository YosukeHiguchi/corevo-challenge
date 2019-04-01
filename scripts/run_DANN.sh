#!/usr/bin/bash

stage=1
model_name=test
gpu=-1

# Network
epoch=50
batchsize=256
optimizer='Adam'
lam_max=0.0

# data
feature='mfcc'

. utils/parse_options.sh

if [ $stage -le 1 ]; then
    python -W ignore run_DANN.py --train \
            --gpu $gpu \
            --epoch $epoch \
            --batchsize $batchsize \
            --optimizer $optimizer \
            --feature $feature \
            --out DANN/model/$model_name \
            --lam_max $lam_max
fi

if [ $stage -le 2 ]; then
    python -W ignore run_DANN.py --predict \
            --gpu $gpu \
            --enc_path DANN/model/$model_name/${epoch}.enc \
            --lc_path DANN/model/$model_name/${epoch}.lc \
            --feature $feature \
            --out DANN/result/$model_name
fi
