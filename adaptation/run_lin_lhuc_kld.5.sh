#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

[ -f path.sh ] && . ./path.sh
[ -f cmd.sh ] && . ./cmd.sh

stage=4

all_speaker="Hubei-C0-S7001 Hubei-C0-S7002 Hubei-C0-S7003 Hubei-C0-S7004
             Hubei-C0-S7005 Hubei-C0-S7006 Hubei-C0-S7007 Hubei-C0-S7008
             Hubei-C0-S7009 Hubei-C0-S7010"
# all_speaker="Hubei-C0-S7001 Hubei-C0-S7002 Hubei-C0-S7003"
# all_speaker="Hubei-C0-S7003 Hubei-C0-S7004"
# all_speaker="xiaomi-asr-data-001-G4023 xiaomi-asr-data-001-G4028"

adaptation_data_dir=data/adaptation_hubei
utt_num=6
lin_lhuc_kld_dir=exp/nnet3/tdnn_lstm1a_lin_lhuc_kld
decode_adapt_dir=$lin_lhuc_kld_dir/decode_adapt_hubei_lin_lhuc_kld_${utt_num}
ivectors_adapt_test=ivectors_adapt_hubei

speaker_file=$adaptation_data_dir/speaker
train_sets_file=$adaptation_data_dir/train_sets
lang=data/lang
am_dir=exp/nnet3/tdnn_lstm1a
edits_conf=conf/edits_lhuc.conf

nnet3_affix=
gmm_dir=exp/tri3
graph_dir=$gmm_dir/graph_adapt

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM options
train_stage=0
label_delay=5

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0

# training options
srand=0
remove_egs=true

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# Insert LHUC layer
if [ $stage -le -1 ]; then
  mkdir -p $lin_lhuc_kld_dir || exit 1;
  echo "======================================================================"
  echo "               Insert LHUC Layer and Initialize                        "
  echo "======================================================================"

  nnet3-am-copy --binary=false  $am_dir/final.mdl $lin_lhuc_kld_dir/si_am.mdl
  nnet3-am-copy --binary=false --edits-config=$edits_conf \
    $lin_lhuc_kld_dir/si_am.mdl $lin_lhuc_kld_dir/fixed_am.mdl

  # LIN layer
  misc/insert_lin_layer.py --lda_dim=300 --nnet_in=$lin_lhuc_kld_dir/fixed_am.mdl \
    --nnet_out=$lin_lhuc_kld_dir/lin.mdl
  # LHUC layer
  python misc/insert_lhuc_layer.py  --nnet-in=$lin_lhuc_kld_dir/lin.mdl \
    --nnet-out=$lin_lhuc_kld_dir/insert.mdl

  nnet3-am-copy --binary=true $lin_lhuc_kld_dir/insert.mdl $lin_lhuc_kld_dir/lin_lhuc_kld.mdl
fi

# Extract ivector for adaptation train data
echo -n "" > $train_sets_file || exit 1;
cat $speaker_file | while read line; do
  echo -n "$line " >> $train_sets_file
done

test_sets=$(cat $train_sets_file)
if [ $stage -le 0 ]; then
  echo "======================================================================"
  echo "                          Extract iVector                             "
  echo "======================================================================"
  if [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ]; then
    rm exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error || exit 1;
  fi

  for data in $test_sets; do
  (
    nspk=$(wc -l < $adaptation_data_dir/${data}_hires/test/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj "$nspk" $adaptation_data_dir/${data}_hires/test \
      exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires
  ) || touch exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error &
  done
  wait
  [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ] && \
    echo "$0: there was a problem while extracting test ivector" && exit 1;
fi

train_sets=$(cat $train_sets_file)
if [ $stage -le 1 ]; then
  echo "======================================================================"
  echo "                          Extract iVector                             "
  echo "======================================================================"
  if [ -f exp/nnet3${nnet3_affix}/ivectors_adapt_train/.error ]; then
    rm exp/nnet3${nnet3_affix}/ivectors_adapt_train/.error || exit 1;
  fi

  for data in $train_sets; do
  (
    nspk=$(wc -l < $adaptation_data_dir/${data}_hires/train/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj "$nspk" $adaptation_data_dir/${data}_hires/train \
      exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_adapt_train/${data}_hires
  ) || touch exp/nnet3${nnet3_affix}/ivectors_adapt_train/.error &
  done
  wait
  [ -f exp/nnet3${nnet3_affix}/ivectors_adapt_train/.error ] && \
    echo "$0: there was a problem while extracting ivector" && exit 1;
fi

ali_dir=exp/nnet3${nnet3_affix}_ali_adapt
# Force alignment
if [ $stage -le 2 ]; then
  echo "======================================================================"
  echo "                      Doing Force Alignment                           "
  echo "======================================================================"
  if [ -f $ali_dir/.error ]; then
    rm $ali_dir/.error || exit 1;
  fi
  for data in $train_sets; do
  (
    if [ -f $ali_dir/$data/ali.1.gz ]; then
      echo "$0: alignments in $ali_dir/$data appear to already exist."
      echo "Please either remove them or use a later --stage option."
      exit 1
    fi
    echo "$0: aligning with the hign-resolution data."

    # nj must is 1
    steps/nnet3/align.sh --use_gpu "false" \
      --cmd "$train_cmd" --nj 1 \
      --online-ivector-dir "exp/nnet3${nnet3_affix}/ivectors_adapt_train/${data}_hires" \
      $adaptation_data_dir/${data}_hires/train $lang $am_dir $ali_dir/$data || exit 1;
  ) || touch $ali_dir/.error &
  done
  wait
  [ -f $ali_dir/.error ] && \
    echo "$0: there was a problem while doing alignment" && exit 1;
fi

# Fine-tune Neural Network
if [ $stage -le 3 ]; then
  echo "======================================================================"
  echo "                         NNet Training                                "
  echo "======================================================================"
  if [ -f exp/.error ]; then
    rm -f exp/.error || exit 1;
  fi
  for train_speaker in $all_speaker; do
  (
    train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_adapt_train/${train_speaker}_hires
    train_data_dir=$adaptation_data_dir/${train_speaker}_hires/train_$utt_num
    dir=$lin_lhuc_kld_dir/lin_lhuc_kld_${train_speaker}_$utt_num
    adapt_ali_dir=$ali_dir/$train_speaker

    model_left_context=$(cat $am_dir/configs/vars | grep "model_left_context" | cut -d "=" -f2)
    model_right_context=$(cat $am_dir/configs/vars | grep "model_right_context" | cut -d "=" -f2)
    left_context=$[$chunk_left_context + $model_left_context]
    right_context=$[$chunk_right_context + $model_right_context]

    samples_per_iter=10000
    num_utts_subsets=1
    # Generate egs for training speaker
    steps/nnet3/get_egs.sh --cmd "$train_cmd" \
      --cmvn-opts "--norm-means=false --norm-vars=false" \
      --transform-dir $adapt_ali_dir \
      --online-ivector-dir $train_ivector_dir \
      --left-context $left_context \
      --right-context $right_context \
      --left-context-initial $model_left_context \
      --right-context-final $model_right_context \
      --stage 0 --nj 1 \
      --num_utts_subset $num_utts_subsets \
      --samples-per-iter $samples_per_iter \
      --frames-per-eg $chunk_width \
      --srand $srand \
      $train_data_dir $adapt_ali_dir $train_data_dir/egs_lin_lhuc_kld_${utt_num}

    if [ -d $dir ]; then
      rm -rf $dir/* || exit 1;
    fi
    mkdir -p $dir/configs || exit 1;
    cp $am_dir/configs/vars $dir/configs || exit 1;
    cp $lin_lhuc_kld_dir/lin_lhuc_kld.mdl $dir/0.mdl || exit 1;
    cp $am_dir/final.mdl $dir/si.mdl || exit 1;

    steps_kld/nnet3/train_rnn.py --stage=$train_stage \
      --cmd="$decode_cmd" \
      --feat.online-ivector-dir=$train_ivector_dir \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --trainer.kld-rho=0.125 \
      --trainer.si-model="$dir/si.mdl" \
      --trainer.srand=$srand \
      --trainer.max-param-change=2.0 \
      --trainer.num-epochs=40 \
      --trainer.deriv-truncate-margin=10 \
      --trainer.samples-per-iter=$samples_per_iter \
      --trainer.optimization.do-final-combination="false" \
      --trainer.optimization.num-jobs-initial=1 \
      --trainer.optimization.num-jobs-final=1 \
      --trainer.optimization.initial-effective-lrate=1e-5 \
      --trainer.optimization.final-effective-lrate=1e-6 \
      --trainer.optimization.shrink-value 1.0 \
      --trainer.rnn.num-chunk-per-minibatch=2:32,64,128 \
      --trainer.optimization.momentum=0.5 \
      --egs.cmd="$train_cmd" \
      --egs.chunk-width=$chunk_width \
      --egs.chunk-left-context=$chunk_left_context \
      --egs.chunk-right-context=$chunk_right_context \
      --egs.chunk-left-context-initial=0 \
      --egs.chunk-right-context-final=0 \
      --egs.dir="$train_data_dir/egs_lin_lhuc_kld_${utt_num}" \
      --cleanup="false" \
      --cleanup.remove-egs=$remove_egs \
      --use-gpu=true \
      --feat-dir=$train_data_dir \
      --ali-dir=$adapt_ali_dir \
      --lang=$lang \
      --reporting.email="$reporting_email" \
      --dir=$dir  || exit 1;
    echo "LOG (LHUC Adaptation): Do adaptation for $train_speaker sucessed."
  ) || touch exp/.error &
  done
  wait
  [ -f exp/.error ] && echo "$0: there was a problem while training" && exit 1;
fi

acwt=0.15    # default is 0.1
# Decode
if [ $stage -le 4 ]; then
  echo "======================================================================"
  echo "                            Decoding                                  "
  echo "======================================================================"
  # test_sets=${train_sets}
  test_sets=$all_speaker
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  for data in ${test_sets}; do
  (
    decode_dir=$decode_adapt_dir/${data}_$utt_num
    mkdir -p $decode_dir || exit 1;
    # cp $lin_lhuc_kld_dir/lin_lhuc_kld_${data}_$utt_num/final.mdl $decode_dir/final.mdl || exit 1;
    cp $lin_lhuc_kld_dir/lin_lhuc_kld_${data}_$utt_num/7.mdl $decode_dir/final.mdl || exit 1;
    cp $am_dir/cmvn_opts $decode_dir || exit 1;
    if [ -f $decode_adapt_dir/.error ]; then
      rm -f $decode_adapt_dir/.error || exit 1;
    fi
    nj=$(wc -l < $adaptation_data_dir/${data}_hires/test/spk2utt)
    steps/nnet3/decode.sh \
      --acwt $acwt \
      --extra-left-context $chunk_left_context \
      --extra-right-context $chunk_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nj --cmd "$decode_bigmem_cmd"  --num-threads 1 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires \
      $graph_dir $adaptation_data_dir/${data}_hires/test \
      $decode_dir/decode_${data} || exit 1;
    steps/scoring/score_kaldi_cer.sh --cmd "$train_cmd" \
      $adaptation_data_dir/${data}_hires/test $graph_dir \
      $decode_dir/decode_${data} || exit 1;
  ) || touch $decode_adapt_dir/.error &
  done
  wait
  [ -f $decode_adapt_dir/.error ] && echo "$0: there was a problem while decoding" && exit 1;
  echo "LOG (LHUC-KLD Decode): Compute CER sucessed."
fi
exit 0;

test_sets=$(cat $train_sets_file)
ivectors_adapt_test=ivectors_adapt_hubei_100
if [ $stage -le 5 ]; then
  echo "======================================================================"
  echo "                          Extract iVector                             "
  echo "======================================================================"
  if [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ]; then
    rm exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error || exit 1;
  fi

  for data in $test_sets; do
  (
    nspk=$(wc -l < $adaptation_data_dir/${data}_hires/test_100/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj "$nspk" $adaptation_data_dir/${data}_hires/test_100 \
      exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires
  ) || touch exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error &
  done
  wait
  [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ] && \
    echo "$0: there was a problem while extracting test ivector" && exit 1;
fi

acwt=0.15    # default is 0.1
decode_adapt_dir=$lin_lhuc_kld_dir/decode_adapt_hubei_lin_lhuc_kld_100_${utt_num}
# Decode
if [ $stage -le 6 ]; then
  echo "======================================================================"
  echo "                            Decoding                                  "
  echo "======================================================================"
  # test_sets=${train_sets}
  test_sets=$all_speaker
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  for data in ${test_sets}; do
  (
    decode_dir=$decode_adapt_dir/${data}_$utt_num
    mkdir -p $decode_dir || exit 1;
    # cp $lin_lhuc_kld_dir/lin_lhuc_kld_${data}_$utt_num/final.mdl $decode_dir/final.mdl || exit 1;
    cp $lin_lhuc_kld_dir/lin_lhuc_kld_${data}_$utt_num/7.mdl $decode_dir/final.mdl || exit 1;
    cp $am_dir/cmvn_opts $decode_dir || exit 1;
    if [ -f $decode_adapt_dir/.error ]; then
      rm -f $decode_adapt_dir/.error || exit 1;
    fi
    nj=$(wc -l < $adaptation_data_dir/${data}_hires/test_100/spk2utt)
    steps/nnet3/decode.sh \
      --acwt $acwt \
      --extra-left-context $chunk_left_context \
      --extra-right-context $chunk_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nj --cmd "$decode_bigmem_cmd"  --num-threads 1 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires \
      $graph_dir $adaptation_data_dir/${data}_hires/test_100 \
      $decode_dir/decode_${data} || exit 1;
    steps/scoring/score_kaldi_cer.sh --cmd "$train_cmd" \
      $adaptation_data_dir/${data}_hires/test_100 $graph_dir \
      $decode_dir/decode_${data} || exit 1;
  ) || touch $decode_adapt_dir/.error &
  done
  wait
  [ -f $decode_adapt_dir/.error ] && echo "$0: there was a problem while decoding" && exit 1;
  echo "LOG (LHUC-KLD Decode): Compute CER sucessed."
fi
exit 0;

