# adopted from prepare_bi.sh

SRC=$1
TGT=$2
MBART_SRC=$3
MBART_TGT=$4
TRAIN_DIR=$5
TRAIN_SET=$6
DEV_SET=$7
DEVTEST_SET=$8

if [ "$FAIRSEQ" = "" ]; then
  echo "you must set environment variable FAIRSEQ to fairseq github directory!"
  exit
fi

ROOT="$(cd "$(dirname -- "$0")/.."; pwd -P)"
MBART=$ROOT/mbart/mbart.cc25.v2
SCRIPTS=$ROOT/spm-scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

DATA=$TRAIN_DIR/data
DATABIN=$TRAIN_DIR/data-bin
mkdir -p $DATA $DATABIN
mkdir $TRAIN_DIR/checkpoints

cp $DEV_SET.$SRC $DATA/dev.$MBART_SRC
cp $DEV_SET.$TGT $DATA/dev.$MBART_TGT
cp $DEVTEST_SET.$SRC $DATA/devtest.$MBART_SRC
cp $DEVTEST_SET.$TGT $DATA/devtest.$MBART_TGT
cp $TRAIN_SET.$SRC $DATA/train.$MBART_SRC
cp $TRAIN_SET.$TGT $DATA/train.$MBART_TGT

# encode train/dv/test
python $SPM_ENCODE \
  --model $MBART/sentence.bpe.model \
  --inputs $DATA/train.$MBART_SRC \
  --outputs $DATA/train.bpe.$MBART_SRC
python $SPM_ENCODE \
  --model $MBART/sentence.bpe.model \
  --inputs $DATA/train.$MBART_TGT \
  --outputs $DATA/train.bpe.$MBART_TGT
for SPLIT in "dev" "devtest"; do \
  python $SPM_ENCODE \
    --model $MBART/sentence.bpe.model \
    --inputs $DATA/$SPLIT.$MBART_SRC \
    --outputs $DATA/$SPLIT.bpe.$MBART_SRC
done
for SPLIT in "dev" "devtest"; do \
  python $SPM_ENCODE \
    --model $MBART/sentence.bpe.model \
    --inputs $DATA/$SPLIT.$MBART_TGT \
    --outputs $DATA/$SPLIT.bpe.$MBART_TGT
done

cat train.bpe.* | sed 's/ /\n/g' | sort | uniq | sed 's/$/ 1/g' > $DATA/trimmed_dict.txt
python $MBART/trim_mbart.py \
  --pre-train-dir $MBART \
  --ft-dict $DATA/trimmed_dict.txt \
  --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
  --output $TRAIN_DIR/checkpoints/trimmed_model.pt

# binarize data
python $FAIRSEQ/fairseq_cli/preprocess.py \
  --source-lang $MBART_SRC --target-lang $MBART_TGT \
  --trainpref $DATA/train.bpe --validpref $DATA/dev.bpe --testpref $DATA/devtest.bpe \
  --destdir $DATABIN \
  --srcdict $DATA/trimmed_dict.txt --tgtdict $DATA/trimmed_dict.txt \
  --workers 8
