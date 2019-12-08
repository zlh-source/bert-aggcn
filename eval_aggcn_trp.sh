#!/bin/bash

SAVE_DIR=$saved_models/01
!python3 eval.py $SAVE_DIR --data_dir dataset/trp/data --dataset test