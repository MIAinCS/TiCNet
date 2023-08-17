#!/bin/bash

set -e

cd ../

if [ -z "$1" ]
    then
    echo "No argument supplied"
    exit 1
fi

exp_name=ticnet
iFold=$1
train_set_name=split/${iFold}_train.csv
val_set_name=split/${iFold}_val.csv
test_set_name=split/${iFold}_val.csv
out_dir=results/${exp_name}/${iFold}_fold

# # Training
python train.py --batch-size 4 --epochs 120 --epoch-rcnn 65 --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $out_dir
