# Cross-Attention Transfer for Machine Translation
This repo hosts the code to accompany the camera-ready version of ["Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation"](https://arxiv.org/abs/2104.08771) in EMNLP 2021.

## Setup
We provide our scripts and modifications to [Fairseq](https://github.com/pytorch/fairseq). In this section, we describe how to go about running the code and, for instance, reproduce Table 2 in the paper.

### Data
To view the data as we prepared and used it, switch to the `main` branch. But we recommend cloning code from this branch to avoid downloading a large amount of data at once. You can always obtain any data as necessary from the `main` branch.

### Installations
We worked in a conda environment with Python 3.8.

* First install the requirements.
  ```bash
    pip install requirements.txt
  ```
* Then install Fairseq. To have the option to modify the package, install it in editable mode.
  ```bash
    cd fairseq-modified
    pip install -e .
  ```
* Finally, set the following environment variable.
  ```bash
    export FAIRSEQ=$PWD
    cd ..
  ```

### Experiments
For the purpose of this walk-through, we assume we want to train a De–En model, using the following data:
  ```bash
  De-En
  ├── iwslt13.test.de
  ├── iwslt13.test.en
  ├── iwslt13.test.tok.de
  ├── iwslt13.test.tok.en
  ├── iwslt15.tune.de
  ├── iwslt15.tune.en
  ├── iwslt15.tune.tok.de
  ├── iwslt15.tune.tok.en
  ├── iwslt16.train.de
  ├── iwslt16.train.en
  ├── iwslt16.train.tok.de
  └── iwslt16.train.tok.en
  ```
by transferring from a Fr–En parent model, the experiment files of which is stored under `FrEn/checkpoints`.

* Start by making an experiment folder and preprocessing the data.
  ```bash
    mkdir test_exp
    ./xattn-transfer-for-mt/scripts/data_preprocessing/prepare_bi.sh \
        de en test_exp/ \
        De-En/iwslt16.train.tok De-En/iwslt15.tune.tok De-En/iwslt13.test.tok \
        8000
  ```
  Please note that `prepare_bi.sh` is written for the most general case, where you are learning vocabulary for both the source and target sides. When necessary       modify it, and reuse whatever vocabulary you want. In this case, e.g., since we are transferring from Fr–En to De–En, we will reuse the target side vocabulary       from the parent. So `8000` refers to the source vocabulary size, and we need to copy parent target vocabulary instead of learning one _in the script_.
  ```bash
    cp ./FrEn/data/tgt.sentencepiece.bpe.model $DATA
    cp ./FrEn/data/tgt.sentencepiece.bpe.vocab $DATA
  ```
* Now you can run an experiment. Here we want to just update the source embeddings and the cross-attention. So we run the corresponding script. Script names are       self-explanatory. Set the correct path to the desired parent model checkpoint _in the script_, and:
  ```bash
    bash ./xattn-transfer-for-mt/scripts/training/reinit-src-embeddings-and-finetune-parent-model-on-translation_src+xattn.sh \
        test_exp/ de en
  ```
* Finally, after training, evaluate your model. Set the correct path to the detokenizer that you use _in the script_, and:
  ```bash
    bash ./xattn-transfer-for-mt/scripts/evaluation/decode_and_score_valid_and_test.sh \
        test_exp/ de en \
        $PWD/De-En/iwslt15.tune.en $PWD/De-En/iwslt13.test.en
  ```

## Issues
Please contact us and report any problems you might face through the issues tab of the repo. Thanks in advance for helping us improve the repo!

## Credits
The main body of code is built upon [Fairseq](https://github.com/pytorch/fairseq). We found it very easy to navigate and modify. Kudos to the developers!  
The data preprocessing scripts are adopted from [FLORES](https://github.com/facebookresearch/flores) scripts.  
To have mBART fit on the GPUs that we worked with memory-wise, we used the trimming solution provided [here](https://github.com/pytorch/fairseq/issues/2120#issuecomment-647429120).

## Citation
```bibtex
@inproceedings{gheini-cross-attention,
  title={Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation},
  author={Gheini, Mozhdeh and Ren, Xiang and May, Jonathan},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021},
  month={November}
}
```
