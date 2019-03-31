#!/bin/bash

# Copyright 2017    Ke Wang

# Begin config
stage=5
train_nj=100
decode_nj=50
# End config

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


train_dir=data/train
test_dir=data/test_50speaker


if [ $stage -le 0 ]; then
  echo ========================================================================
  echo "                Data & Lexicon & Language Preparation                 "
  echo ========================================================================
  # Data Preparation
  lexicon=/home/train03/data/resources/zhongwen.lex
  lm_arpa=/home/train03/data/resources/lmprune/1e_8/lm.upper.arpa.gz

  utils/data/validate_data_dir.sh --no-wav $train_dir || exit 1;
  utils/data/validate_data_dir.sh --no-wav $test_dir || exit 1;
  misc/prep_train_dict.sh $lexicon $train_dir $test_dir || exit 1;
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;

  # prepare lang dir for test
  misc/prep_test_dict.sh $lexicon data/local/dict data/local/dict_test || exit 1;
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt data/local/dict_test \
    "<UNK>" data/local/lang_test_tmp data/lang_test_tmp || exit 1;

  # Change the LM vocabulary to be the intersection of the current LM vocab and the set
  # of words in the pronunciation lexicon. Also renormalizes the LM by recomputing the
  # backoff weights, and remove those ngrams whose probabilities are lower than the
  # backed-off estimates.
  utils/format_lm_sri.sh --srilm_opts "-subset -prune-lowprobs -unk" \
    data/lang_test_tmp $lm_arpa data/lang_test || exit 1;
fi

if [ $stage -le 1 ]; then
  echo ========================================================================
  echo "                          MonoPhone Training                          "
  echo ========================================================================
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/mono
fi

if [ $stage -le 2 ]; then
  echo ========================================================================
  echo "           tri1 : Deltas + Delta-Deltas Training & Decoding           "
  echo ========================================================================
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    7000 130000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
  # Decode tri1
  (
    utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
    steps/decode.sh --cmd "$decode_bigmem_cmd" --config conf/decode.config --nj $decode_nj \
      exp/tri1/graph $test_dir exp/tri1/decode_test || exit 1;
  ) &
fi

if [ $stage -le 3 ]; then
  echo ========================================================================
  echo "           tri2 : Deltas + Delta-Deltas Training & Decoding           "
  echo ========================================================================
  # Align tri1
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    7000 130000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
  # Decode tri2
  (
    utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
    steps/decode.sh --cmd "$decode_bigmem_cmd" --config conf/decode.config --nj $decode_nj \
      exp/tri2/graph $test_dir exp/tri2/decode_test || exit 1;
  ) &
fi

if [ $stage -le 4 ]; then
  echo ========================================================================
  echo "                tri3 : LDA + MLLT Training & Decoding                 "
  echo ========================================================================
  # Triphone wiht LDA+MLLT
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    7000 130000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1;
  # Decode tri3
  (
    utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
    steps/decode.sh --cmd "$decode_bigmem_cmd" --config conf/decode.config --nj $decode_nj \
      exp/tri3/graph $test_dir exp/tri3/decode_test || exit 1;
  ) &
fi

if [ $stage -le 5 ]; then
  echo ========================================================================
  echo "                       NN Training & Decoding                         "
  echo ========================================================================
  # TDNN + LSTM training
  misc/run_tdnn_lstm.sh --nj 100 --gmm "tri3" --stage -10 \
    --train_set "train" --test_sets "test_50speaker" || exit 1;
fi
