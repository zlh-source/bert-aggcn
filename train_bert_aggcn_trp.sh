#!/bin/bash

SAVE_ID=$1
python train.py --data_dir dataset/trp/data --vocab_dir dataset/trp/vocab --id $SAVE_ID --seed 0 --hidden_dim 768 --lr 0.7 --no-rnn  --rnn_hidden 768 --num_epoch 100 --pooling max  --mlp_layers 1 --num_layers 2 --pooling_l2 0.002