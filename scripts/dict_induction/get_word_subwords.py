import argparse


def main():
    parser = argparse.ArgumentParser(description='sift out the subwords that are actual words.')

    parser.add_argument('-i', type=str, help='path to segmented corpus')
    parser.add_argument('-o', type=str, help='path to write the whole-word subwords to')

    args = parser.parse_args()

    sifted_subwords = set()

    with open(args.i) as corpus:
        for line in corpus:
            segments = line.strip().split()
            for i, s in enumerate(segments):
                if i == len(segments) - 1 and s[0] == '▁':
                    sifted_subwords.add(s)
                elif s[0] == '▁' and segments[i+1][0] == '▁':
                    sifted_subwords.add(s)

    print(len(sifted_subwords))

    with open(args.o, 'w') as out:
        for s in sifted_subwords:
            out.write(f'{s}\n')


if __name__ == '__main__':
    main()
