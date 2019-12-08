#!/bin/bash

SAVE_DIR=$saved_models/03
!python3 eval.py $SAVE_DIR --data_dir dataset/pp/data --dataset test