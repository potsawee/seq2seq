#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 1 ) then
   echo "Usage: $0 log"
   echo "  e.g: $0 LOGs/translate.log.txt"
   exit 100
endif

set LOG=$1
set NMTTRANSLATE=/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/run/nmt-translate.sh

set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' $NMTTRANSLATE`
echo $CMD
