# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# modified as necessary

SRC=$1
TGT=$2
TRAIN_DIR=$3
TRAIN_SET=$4
DEV_SET=$5
DEVTEST_SET=$6

BPESIZE=$7

if [ "$FAIRSEQ" = "" ]; then
  echo "you must set environment variable FAIRSEQ to fairseq github directory!"
  exit
fi

ROOT="$(cd "$(dirname -- "$0")/.."; pwd -P)"
SCRIPTS=$ROOT/spm-scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

DATA=$TRAIN_DIR/data
DATABIN=$TRAIN_DIR/data-bin
mkdir -p $DATA $DATABIN

cp $DEV_SET.$SRC $DATA/dev.$SRC
cp $DEV_SET.$TGT $DATA/dev.$TGT
cp $DEVTEST_SET.$SRC $DATA/devtest.$SRC
cp $DEVTEST_SET.$TGT $DATA/devtest.$TGT
cp $TRAIN_SET.$SRC $DATA/train.$SRC
cp $TRAIN_SET.$TGT $DATA/train.$TGT

# learn BPE with sentencepiece
# note that if transferring from a parent, we reuse parent's .model and .vocab files
# for the side that remains the same. So respective bpe learning segment can be skipped.
# example: when transferring from Fr-En to Ro-En, we reuse tgt.sentencepiece.bpe.{model,vocab}
# from Fr-En model.
python $SPM_TRAIN \
 --input=$DATA/train.$SRC \
 --model_prefix=$DATA/src.sentencepiece.bpe \
 --vocab_size=$BPESIZE \
 --model_type=bpe \
 --input_sentence_size=15000000 \
 --pad_id=3

python $SPM_TRAIN \
  --input=$DATA/train.$TGT \
  --model_prefix=$DATA/tgt.sentencepiece.bpe \
  --vocab_size=$BPESIZE \
  --model_type=bpe \
  --input_sentence_size=15000000 \
  --pad_id=3

# encode train/dv/test
python $SPM_ENCODE \
  --model $DATA/src.sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $DATA/train.$SRC \
  --outputs $DATA/train.bpe.$SRC
python $SPM_ENCODE \
  --model $DATA/tgt.sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $DATA/train.$TGT \
  --outputs $DATA/train.bpe.$TGT
for SPLIT in "dev" "devtest"; do \
  python $SPM_ENCODE \
    --model $DATA/src.sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $DATA/$SPLIT.$SRC \
    --outputs $DATA/$SPLIT.bpe.$SRC
done
for SPLIT in "dev" "devtest"; do \
  python $SPM_ENCODE \
    --model $DATA/tgt.sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $DATA/$SPLIT.$TGT \
    --outputs $DATA/$SPLIT.bpe.$TGT
done

cut -f1 $DATA/src.sentencepiece.bpe.vocab | tail -n +5 | sed "s/$/ 100/g" > $DATA/src.fairseq.bpe.vocab
cut -f1 $DATA/tgt.sentencepiece.bpe.vocab | tail -n +5 | sed "s/$/ 100/g" > $DATA/tgt.fairseq.bpe.vocab

# binarize data
python $FAIRSEQ/fairseq_cli/preprocess.py \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $DATA/train.bpe --validpref $DATA/dev.bpe --testpref $DATA/devtest.bpe \
  --destdir $DATABIN \
  --srcdict $DATA/src.fairseq.bpe.vocab --tgtdict $DATA/tgt.fairseq.bpe.vocab \
  --workers 16
