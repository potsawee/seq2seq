#!/bin/bash
# souce = load the conda environment that includes
# python 3.6 bin path
# TensorFlow (GPU) 1.5.0
source /home/miproj/urop.2018/pm574/anaconda3/bin/activate tf_gpu

# python 3.6 bin path
# TensorFlow (CPU) 1.10.0
# source /home/miproj/urop.2018/pm574/anaconda3/bin/activate tf_cpu

python /home/alta/BLTSpeaking/ged-pm574/local/seq2seq/train.py
