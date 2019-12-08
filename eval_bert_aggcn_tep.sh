#!/bin/bash

SAVE_DIR=$saved_models/02
!python3 eval.py $SAVE_DIR --data_dir dataset/tep/data --dataset test