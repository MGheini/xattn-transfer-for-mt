DIR=$1
cd $DIR

SRC=$2
TGT=$3

# SET THE PATH TO THE PARENT MODEL
PARENT=../FrEn/checkpoints/checkpoint_best.pt

python $FAIRSEQ/fairseq_cli/train.py data-bin \
	--source-lang $SRC --target-lang $TGT \
	--log-format simple \
	--log-interval 20 \
	--seed 1 \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--optimizer adam \
	--adam-eps 1e-06 \
	--adam-betas "(0.9, 0.98)" \
	--weight-decay 0.0 \
	--lr-scheduler polynomial_decay \
	--task translation \
	--eval-bleu --eval-bleu-detok moses \
	--num-workers 8 \
	--max-tokens 2048 \
	--validate-interval 1 \
	--arch transformer \
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
	--dropout 0.1 \
	--attention-dropout 0.1 \
	--relu-dropout 0.0 \
	--layernorm-embedding \
	--encoder-learned-pos \
	--decoder-learned-pos \
	--no-scale-embedding \
	--encoder-normalize-before \
	--decoder-normalize-before \
	--skip-invalid-size-inputs-valid-test \
	--share-decoder-input-output-embed \
	--load-model-but-src-embeddings-and-freeze-tgt-embeddings-from $PARENT \
	--only-finetune-cross-attn \
	--patience 25
