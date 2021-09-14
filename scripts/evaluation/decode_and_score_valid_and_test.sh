DIR=$1
SRC=$2
TGT=$3
VALIDREF=$4
TESTREF=$5

# SET YOUR OWN DETOKENIZER PATH
DETOKENIZER=~/scripts/detokenizer.perl

cd $DIR

python $FAIRSEQ/fairseq_cli/generate.py data-bin/ \
	--gen-subset valid \
	--task translation \
	-s $SRC -t $TGT \
	--path checkpoints/checkpoint_best.pt \
	--beam 5 --lenpen 1.2 \
	--remove-bpe sentencepiece \
	--results-path decodes/ \
	--sacrebleu --max-tokens 12000

python $FAIRSEQ/fairseq_cli/generate.py data-bin/ \
	--gen-subset test \
	--task translation \
	-s $SRC -t $TGT \
	--path checkpoints/checkpoint_best.pt \
	--beam 5 --lenpen 1.2 \
	--remove-bpe sentencepiece \
	--results-path decodes/ \
	--sacrebleu --max-tokens 12000

echo $PWD
cd decodes/

grep "^H-[0-9]*" generate-valid.txt | sed 's/^H-//g' | sort -k1 -n | cut -f3 > tune.out.tok
grep "^H-[0-9]*" generate-test.txt  | sed 's/^H-//g' | sort -k1 -n | cut -f3 > test.out.tok

$DETOKENIZER < tune.out.tok > tune.out.tok.detok
$DETOKENIZER < test.out.tok > test.out.tok.detok

sacrebleu $VALIDREF -i tune.out.tok.detok > tune.bleu
sacrebleu $TESTREF  -i test.out.tok.detok > test.bleu
