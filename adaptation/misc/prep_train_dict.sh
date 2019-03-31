#!/bin/bash

# Copyright 2017    Jian Li  Xiaomi
#                   Ke Wang

[ $# != 3 ] && echo "Usage: `basename $0` <dict-file> <train-dir> <test_dir> " && exit 1;

dict=$1
train_dir=$2
test_dir=$3
dict_dir=data/local/dict

mkdir -p $dict_dir
mkdir -p $dict_dir/lexicon

# extract full vocabulary
cat $train_dir/text $test_dir/text | awk '{for (i = 2; i <= NF; i++) print $i}' | sort -u > $dict_dir/words.txt || exit 1;

## extract in-vocab lexicon and oov words
echo "--- Searching for OOV words ..."
awk 'NR==FNR {dict[$1]; next;} !($1 in dict)' \
  $dict $dict_dir/words.txt |\
  egrep -v '<.?s>' | sort -u > $dict_dir/lexicon/words-oov.txt;

awk 'NR==FNR {words[$1]; next;} ($1 in words)' \
  $dict_dir/words.txt $dict |\
  egrep -v '<.?s>' | sort -u > $dict_dir/lexicon/lexicon-iv.txt || exit 1;

wc -l $dict_dir/lexicon/words-oov.txt
wc -l $dict_dir/lexicon/lexicon-iv.txt

cat $dict_dir/lexicon/lexicon-iv.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  sort -u |\
  perl -e '
  my %ph_cl;
  while (<STDIN>) {
    $phone = $_;
    chomp($phone);
    chomp($_);
    $phone =~ s:([A-Z]+)[0-9]:$1:;
    if (exists $ph_cl{$phone}) { push(@{$ph_cl{$phone}}, $_)  }
    else { $ph_cl{$phone} = [$_]; }
  }
  foreach $key ( keys %ph_cl ) {
     print "@{ $ph_cl{$key} }\n"
  }
  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

( echo SIL; echo SPN; echo NSN ) > $dict_dir/silence_phones.txt

echo SIL > $dict_dir/optional_silence.txt

cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dict_dir/extra_questions.txt || exit 1;

# Add to the lexicon the silences, noises etc.
( echo '!SIL SIL'; echo '<UNK> SPN'; echo '[NOISE] NSN' ) | \
 cat - $dict_dir/lexicon/lexicon-iv.txt  > $dict_dir/lexicon.txt || exit 1;

echo "$0: dict preparation succeeded"

exit 0;
