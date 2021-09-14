DIR=$1
cd $DIR

SRC=$2
TGT=$3

MBART=$4

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

python $FAIRSEQ/fairseq_cli/train.py data-bin \
	--langs $langs \
	--source-lang $SRC --target-lang $TGT \
	--log-format simple \
	--log-interval 20 \
	--seed 222 \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing 0.2 \
	--optimizer adam \
	--adam-eps 1e-06 \
	--adam-betas "(0.9, 0.98)" \
	--weight-decay 0.0 \
	--lr-scheduler polynomial_decay \
	--task translation_from_pretrained_bart \
	--eval-bleu --eval-bleu-detok moses \
	--num-workers 8 \
	--max-tokens 512 \
	--validate-interval 1 \
	--arch mbart_large \
	--max-update 150000 \
	--update-freq 8 \
	--lr 3e-05 \
	--min-lr -1 \
	--restore-file checkpoint_last.pt \
	--save-interval 1 \
	--save-interval-updates 500 \
	--keep-interval-updates 1 \
	--no-epoch-checkpoints \
	--warmup-updates 2500 \
	--dropout 0.3 \
	--attention-dropout 0.1 \
	--relu-dropout 0.0 \
	--layernorm-embedding \
	--encoder-learned-pos \
	--decoder-learned-pos \
	--encoder-normalize-before \
	--decoder-normalize-before \
	--skip-invalid-size-inputs-valid-test \
	--share-all-embeddings \
	--finetune-from-mbart-at $MBART \
	--only-finetune-cross-attn \
	--patience 25
