#!/bin/bash

# Copyright 2017    Ke Wang

# This script is for preparing adaptation data.
# We Select the speaker and ensure every speaker's utterances is more than
# #utt_per_speaker and random select #train_utt_per_speaker as train sets,
# the rest as test sets.

set -euo pipefail

stage=0

speaker_all_dir=/home/dl87/data/dialect/King-ASR-384-17-Hubei/mfcc-40dim
test_speaker_dir=data/test_hubei_hires
speaker_info_file=$test_speaker_dir/speaker_info
adaptation_dir=data/adaptation_hubei
adaptation_speaker=$adaptation_dir/speaker

num_test=20
utt_per_speaker=490
train_utt_per_speaker=450

# Get test speaker id
if [ $stage -le 0 ]; then
  mkdir -p $test_speaker_dir
  echo -n "" > $speaker_info_file
  cat $speaker_all_dir/spk2utt | while read line; do
    utt_id=$(echo $line | awk '{print $1}')
    echo $line > tmp_spk2utt
    utils/spk2utt_to_utt2spk.pl tmp_spk2utt > tmp_utt2spk
    num_utt=$(wc -l tmp_utt2spk | cut -d " " -f1)
    if [ $num_utt -ge $utt_per_speaker ]; then
      echo "$utt_id $num_utt" >> $speaker_info_file
      rm tmp_spk2utt tmp_utt2spk || exit 1;
    fi
  done
  echo "Get all test speaker info done."
fi

# Get test data
if [ $stage -le 1 ]; then
  cp $speaker_all_dir/{feats.scp,spk2utt,utt2spk,text,wav.scp} $test_speaker_dir
  cat $speaker_info_file | awk '{print $1}' > $test_speaker_dir/speaker_id
  misc/select_data.py $test_speaker_dir/speaker_id $speaker_all_dir/cmvn.scp \
    $test_speaker_dir/cmvn.txt
  head -n $num_test $test_speaker_dir/cmvn.txt > $test_speaker_dir/cmvn.scp
  utils/fix_data_dir.sh $test_speaker_dir
  rm -rf $test_speaker_dir/{cmvn.txt,.backup} || exit 1;
  echo "Get $num_test speaker data done."
fi

# Get per speaker data
if [ $stage -le 2 ]; then
  mkdir -p $adaptation_dir || exit 1;
  cat $test_speaker_dir/cmvn.scp | awk '{print $1}' > $adaptation_speaker
  num=0
  cat $adaptation_speaker | while read line; do
    data_dir=$adaptation_dir/${line}_hires
    mkdir -p $data_dir || exit 1;
    cp $test_speaker_dir/{cmvn.scp,feats.scp,spk2utt,utt2spk,text,wav.scp} \
      $data_dir || exit 1;
    grep $line $data_dir/cmvn.scp > $data_dir/cmvn.txt
    mv $data_dir/cmvn.txt $data_dir/cmvn.scp || exit 1;
    utils/fix_data_dir.sh $data_dir > /dev/null || exit 1;
    rm -rf $data_dir/.backup || exit 1;

    # Split train/test sets
    mkdir -p $data_dir/{train,test} || exit 1;
    misc/shuffle_list.py $data_dir/feats.scp $data_dir/feats.txt
    cat $data_dir/feats.txt | head -$train_utt_per_speaker | \
      sort > $data_dir/train/feats.scp || exit 1;
    rm $data_dir/feats.txt || exit 1;
    cat $data_dir/train/feats.scp | awk '{print $1}' > $data_dir/train_utt_id
    utils/filter_scp.pl --exclude $data_dir/train_utt_id \
      $data_dir/feats.scp | sort > $data_dir/test/feats.scp || exit 1;
    rm $data_dir/train_utt_id || exit 1;
    cp $data_dir/{cmvn.scp,spk2utt,utt2spk,wav.scp,text} $data_dir/train || exit 1;
    cp $data_dir/{cmvn.scp,spk2utt,utt2spk,wav.scp,text} $data_dir/test || exit 1;
    rm $data_dir/{cmvn.scp,feats.scp,spk2utt,utt2spk,text,wav.scp} || exit 1;
    utils/fix_data_dir.sh $data_dir/train > /dev/null || exit 1;
    utils/fix_data_dir.sh $data_dir/test > /dev/null|| exit 1;
    rm -r $data_dir/{train,test}/.backup || exit 1;
    num=$[num+1]
    echo "No.$num Prepare ${line}_hires"
  done
fi
