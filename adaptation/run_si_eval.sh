#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail
[ -f path.sh ] && . ./path.sh
[ -f cmd.sh ] && . ./cmd.sh

# Begin configuration section
stage=1
decode_threads=1
# End configuration section

nnet3_affix=
gmm_dir=exp/tri3

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0

adaptation_data_dir=data/adaptation_hubei
ivectors_adapt_test=ivectors_adapt_hubei_100
speaker_file=${adaptation_data_dir}/speaker
test_sets_file=${adaptation_data_dir}/test_sets
am_dir=exp/nnet3/tdnn_lstm1a
decode_dir=${am_dir}/adaptation_lm_hubei_100
graph_dir=$gmm_dir/graph_adapt

if [ $stage -le -1 ]; then
  lm_arpa=exp/lm_adapt/lm.adapt.arpa.gz
  utils/format_lm_sri.sh --srilm_opts "-subset -prune-lowprobs -unk" \
    data/lang_test_tmp $lm_arpa data/lang_adapt_test || exit 1;
  utils/mkgraph.sh data/lang_adapt_test exp/tri3 $graph_dir || exit 1;
fi

mkdir -p ${decode_dir} || exit 1;
cp ${am_dir}/{cmvn_opts,final.mdl} ${decode_dir} || exit 1;

echo -n "" > ${test_sets_file} || exit 1;
cat ${speaker_file} | while read line; do
  echo -n "${line} " >> ${test_sets_file}
done

test_sets=$(cat ${test_sets_file})
# Extract i-vectors
if [ $stage -le 0 ]; then
  if [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ]; then
    rm exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error || exit 1;
  fi
  for data in ${test_sets}; do
  (
    nspk=$(wc -l < ${adaptation_data_dir}/${data}_hires/test_100/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj "${nspk}" ${adaptation_data_dir}/${data}_hires/test_100/ \
      exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires
  ) || touch exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error &
  done
  wait
  [ -f exp/nnet3${nnet3_affix}/$ivectors_adapt_test/.error ] && \
    echo "$0: there was a problem while extracting ivector" && exit 1;
fi

min_lmwt=7  # default is 7
max_lmwt=17 # default is 17
acwt=0.15    # default is 0.1
scoring_opts="--min_lmwt $min_lmwt --max_lmwt $max_lmwt"
if [ $stage -le 1 ]; then
  if [ -f ${decode_dir}/.error ]; then
    rm ${decode_dir}/.error || exit 1;
  fi
  for data in ${test_sets}; do
  (
    frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
    nj=$(wc -l < ${adaptation_data_dir}/${data}_hires/test/spk2utt)
    steps/nnet3/decode.sh \
      --scoring_opts "$scoring_opts" \
      --acwt $acwt \
      --extra-left-context $chunk_left_context \
      --extra-right-context $chunk_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nj --cmd "$train_cmd"  --num-threads ${decode_threads} \
      --online-ivector-dir exp/nnet3${nnet3_affix}/$ivectors_adapt_test/${data}_hires \
      ${graph_dir} ${adaptation_data_dir}/${data}_hires/test_100 \
      ${decode_dir}/decode_${data} || exit 1;
    steps/scoring/score_kaldi_cer.sh --cmd "$train_cmd" $scoring_opts\
      ${adaptation_data_dir}/${data}_hires/test_100 ${graph_dir} \
      ${decode_dir}/decode_${data} || exit 1;
  ) || touch ${decode_dir}/.error &
  done
  wait
  [ -f ${decode_dir}/.error ] && echo "$0: there was a problem while decoding" && exit 1;
fi
exit 0;
