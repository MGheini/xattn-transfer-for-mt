import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description='extract embeddings from a trained model.')

    parser.add_argument('-m', type=str, help='path to the model')
    parser.add_argument('-o', type=str, help='path to write the embeddings to')
    parser.add_argument('-t', action='store_true', help='extract tgt embeddings; '
                                                        'if not set, extract src embeddings by default.')

    args = parser.parse_args()

    if not args.t:
        embeddings = torch.load(args.m)['model']['encoder.embed_tokens.weight']
    else:
        embeddings = torch.load(args.m)['model']['decoder.embed_tokens.weight']
    print(embeddings.size())

    with open(args.o, 'w') as output:
        for t in embeddings[4:]:
            v = [round(d, 5) for d in t.numpy()]
            output.write('\t'.join([str(d) for d in v]) + '\n')


if __name__ == '__main__':
    main()
