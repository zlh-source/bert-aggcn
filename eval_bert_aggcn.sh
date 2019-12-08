#!/bin/bash

SAVE_DIR=$saved_models/01
!python3 eval.py $SAVE_DIR --data_dir dataset/trp/data --dataset test

SAVE_DIR=$saved_models/02
!python3 eval.py $SAVE_DIR --data_dir dataset/tep/data --dataset test

SAVE_DIR=$saved_models/03
!python3 eval.py $SAVE_DIR --data_dir dataset/pp/data --dataset test