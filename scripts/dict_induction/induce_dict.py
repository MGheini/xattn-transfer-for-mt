import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='dictionary induction over a vocabulary sample '
                                                 'through nearest neighbours.')

    parser.add_argument('--gold', type=str, help='path to the gold dictionary.')

    parser.add_argument('--l1voc', type=str, help='path to l1 vocab.')
    parser.add_argument('--l1vec', type=str, help='path to l1 vec.')
    parser.add_argument('--l2voc', type=str, help='path to l2 vocab.')
    parser.add_argument('--l2vec', type=str, help='path to l2 vec.')

    args = parser.parse_args()

    l1_dict = {}
    l2_dict = {}
    gold_dict = {}

    with open(args.l1voc) as voc_file, open(args.l1vec) as vec_file:
        for voc, vec in zip(voc_file, vec_file):
            l1_dict[voc.strip()] = np.fromstring(vec.strip(), dtype=float, sep='\t')

    with open(args.l2voc) as voc_file, open(args.l2vec) as vec_file:
        for voc, vec in zip(voc_file, vec_file):
            l2_dict[voc.strip()] = np.fromstring(vec.strip(), dtype=float, sep='\t')

    with open(args.gold) as gold:
        for line in gold:
            gold_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

    evaluation_set = []
    for word in l1_dict:
        if word[1:].lower() in gold_dict:
            evaluation_set.append(word)

    induced_dict = []

    c = 0

    for word in evaluation_set:
        max_sim = -1
        src_voc = word
        translation = ''

        src_vec = l1_dict[src_voc]
        for tgt_voc in l2_dict:
            sim = np.dot(src_vec, l2_dict[tgt_voc])
            if sim > max_sim:
                max_sim = sim
                translation = tgt_voc

        induced_dict.append((src_voc, translation))

        if translation[1:].lower() == gold_dict[src_voc[1:].lower()]:
            c += 1

    print(len(induced_dict))
    print(c)


if __name__ == '__main__':
    main()
