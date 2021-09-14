import copy
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description='build a checkpoint by combining the'
                                                 'source embeddings from one and the body'
                                                 'from another.')

    parser.add_argument('-b', type=str, help='path to the checkpoint to take the body from')
    parser.add_argument('-s', type=str, help='path to the checkpoint to take the source '
                                             'embeddings from')
    parser.add_argument('-o', type=str, help='directory path to write the frankenstein '
                                             'checkpoint to')
    parser.add_argument('-t', '--src-frm-translation-model', action='store_true', default=False)

    args = parser.parse_args()

    body_ckpt = torch.load(args.b)
    src_embeddings_ckpt = torch.load(args.s)
    frankenstein_ckpt = copy.deepcopy(body_ckpt)

    for p in src_embeddings_ckpt['model']:
        if p == 'encoder.embed_tokens.weight' and not args.src_frm_translation_model:
            frankenstein_ckpt['model'][p] = src_embeddings_ckpt['model'][p][:-1, :]
        elif p == 'encoder.embed_tokens.weight':
            frankenstein_ckpt['model'][p] = src_embeddings_ckpt['model'][p]
        elif p in ['encoder.embed_positions.weight',
                 'encoder.layernorm_embedding.weight',
                 'encoder.layernorm_embedding.bias',]:
            frankenstein_ckpt['model'][p] = src_embeddings_ckpt['model'][p]

    torch.save(frankenstein_ckpt, args.o + '/checkpoint_frankenstein.pt')


if __name__ == '__main__':
    main()
