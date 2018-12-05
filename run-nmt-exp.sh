#!/bin/bash

qsub -cwd -j yes -o LOGs/train.log \
    -P esol \
    -l qp=cuda_low \
    -l osrel='*' \
    -l mem_grab=60G -l mem_free=60G \
    nmt-exp.sh
