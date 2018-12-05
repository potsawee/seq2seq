#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 1 ) then
   echo "Usage: $0 log"
   echo "  e.g: $0 LOGs/nmt-en1.train.txt"
   exit 100
endif

set LOG=$1
set NMTTRAIN=/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/nmt-train.sh

set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda_low -l osrel='*' -l mem_grab=60G -l mem_free=60G  $NMTTRAIN`
echo $CMD
