# export KALDI_ROOT=/home/train06/wangke/tools/kaldi
export KALDI_ROOT=/home/dl87/wangke/tools/kaldi-wangke
# export KALDI_ROOT=/home/wangke/tools/kaldi-wangke
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
