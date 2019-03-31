#!/bin/bash

# Copyright 2017    Ke Wang

# This script is to preprape differen time durations train sets for adapatatio
# training.

set -euo pipefail

if [ -f path.sh ]; then . ./path.sh; fi

duration="100"

adaptation_data_dir=data/adaptation_hubei
speaker_file=${adaptation_data_dir}/speaker
test_sets_file=${adaptation_data_dir}/test_sets
test_sets=$(cat ${test_sets_file})

for x in $test_sets; do
  folder=$adaptation_data_dir/${x}_hires
  test_folder=${folder}/test_${duration}
  mkdir -p $test_folder || exit 1
  cp -r $folder/train/{feats.scp,cmvn.scp,spk2utt,utt2spk,text} \
    $test_folder || exit 1
  tail -n $duration $test_folder/feats.scp > $test_folder/feats.txt
  mv $test_folder/feats.txt $test_folder/feats.scp || exit 1
  utils/fix_data_dir.sh $test_folder || exit 1
done

exit 0
