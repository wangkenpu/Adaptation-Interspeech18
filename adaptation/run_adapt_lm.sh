#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

train_cmd="run.pl"

[ -f path.sh ] && . ./path.sh

data=data/lm_adapt
script="data001.text hubei.text sichuan.text"
text=$data/text
lm_adapt=exp/lm_adapt/lm.adapt.arpa.gz

# Prepare training text
echo -n "" > $text
for x in $script; do
  cut -d " " -f 2- $data/$x >> $text
done

ngram-count -debug 1 -text $text -order 3 -ukndiscount -interpolate -lm $lm_adapt || exit 1;

echo "Train N-Garms done."
