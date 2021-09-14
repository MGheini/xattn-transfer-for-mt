import copy
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description='build a checkpoint by combining the'
                                                 'source embeddings from one and the body'
                                                 'from another.')

    parser.add_argument('-b', type=str, help='path to the checkpoint to take the body from')
    parser.add_argument('-t', type=str, help='path to the checkpoint to take the target '
                                             'embeddings from')
    parser.add_argument('-o', type=str, help='directory path to write the frankenstein '
                                             'checkpoint to')
    parser.add_argument('-tr', '--tgt-frm-translation-model', action='store_true', default=False)

    args = parser.parse_args()

    body_ckpt = torch.load(args.b)
    tgt_embeddings_ckpt = torch.load(args.t)
    frankenstein_ckpt = copy.deepcopy(body_ckpt)

    for p in tgt_embeddings_ckpt['model']:
        if (p == 'decoder.embed_tokens.weight' or p == 'decoder.output_projection.weight') and not args.tgt_frm_translation_model:
            frankenstein_ckpt['model'][p] = tgt_embeddings_ckpt['model'][p][:-1, :]
        elif (p == 'decoder.embed_tokens.weight' or p == 'decoder.output_projection.weight'):
            frankenstein_ckpt['model'][p] = tgt_embeddings_ckpt['model'][p]
        elif p in ['decoder.embed_positions.weight',
                   'decoder.layernorm_embedding.weight',
                   'decoder.layernorm_embedding.bias',]:
            frankenstein_ckpt['model'][p] = tgt_embeddings_ckpt['model'][p]

    torch.save(frankenstein_ckpt, args.o + '/checkpoint_frankenstein.pt')


if __name__ == '__main__':
    main()
