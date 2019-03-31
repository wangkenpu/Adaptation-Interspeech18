#!/bin/bash

# Copyright 2017    Jian Li  Xiaomi
#                   Ke Wang

# generate new lang dir for test

. ./utils/parse_options.sh

lexicon=$1
old_dict=$2
new_dict=$3

if [ ! -d $old_dict ]; then
  echo "$old_dict dir doesn't exist!" && exit 1;
fi

if [ ! -d data/lang ]; then
  echo "data/lang dir doesn't exist!" && exit 1;
fi

mkdir -p $new_dict/tmp
cp $old_dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} $new_dict

cat $lexicon | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' > $new_dict/tmp/all_phones.txt

cat $new_dict/{nonsilence_phones.txt,silence_phones.txt} > $new_dict/tmp/old_phones.txt

grep -w -F -v -f $new_dict/tmp/old_phones.txt $new_dict/tmp/all_phones.txt > $new_dict/tmp/extra_phones.txt

echo '<UNK> SPN' | \
  cat - $lexicon | grep -w -v -f $new_dict/tmp/extra_phones.txt > $new_dict/lexicon.txt || exit 1;

rm -r $new_dict/tmp || exit 1;

exit 0;
