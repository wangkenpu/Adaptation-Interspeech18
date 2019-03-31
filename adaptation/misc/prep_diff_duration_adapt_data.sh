#!/bin/bash

# Copyright 2017    Ke Wang

# This script is to preprape differen time durations train sets for adapatatio
# training.

set -euo pipefail

if [ -f path.sh ]; then . ./path.sh; fi

duration="6 11 21 31 51 81 101 151 201 301"

adaptation_data_dir=data/adaptation_hubei
speaker_file=${adaptation_data_dir}/speaker
test_sets_file=${adaptation_data_dir}/test_sets

# test_sets="Hubei-C0-S7001"
echo -n "" > $test_sets_file
cat $speaker_file | while read line; do
  echo -n "$line " >> $test_sets_file
done
test_sets=$(cat ${test_sets_file})

for x in $test_sets; do
  for dur in $duration; do
    folder=$adaptation_data_dir/${x}_hires/train
    mkdir -p ${folder}_$dur || exit 1
    cp -r $folder/{feats.scp,cmvn.scp,spk2utt,utt2spk,text} ${folder}_$dur || exit 1
    head -n $dur ${folder}_$dur/feats.scp > ${folder}_$dur/feats.txt
    mv ${folder}_$dur/feats.txt ${folder}_$dur/feats.scp || exit 1
    utils/fix_data_dir.sh ${folder}_$dur || exit 1
  done
done

exit 0
