import argparse
import glob


def main(args):

    inputs = glob.glob(args.input_files)
    print('Found %s input files.' % len(inputs))
    num_words = 0
    word_counter = {}
    for filename in inputs:
        print('Processing file %s' % filename)
        with open(filename, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        for sentence in sentences:
            for word in sentence.split():
                num_words += 1
                if word in word_counter.keys():
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1
    
    print('Total words: %s' % num_words)
    print('Total unique words: %s' % len(word_counter))

    cutoff = args.freq_cutoff
    if cutoff > 0:
        word_counter = { k : v for k, v in word_counter.items() if v > cutoff }
    
    if args.add_special_symbols:
        word_counter.pop('<S>', None)
        word_counter.pop('</S>', None)
        word_counter.pop('<UNK>', None)
    word_list = [k for k, v in sorted(word_counter.items(), key=lambda kv : kv[1], reverse=True)]
    if args.max_vocab_size is not None and args.max_vocab_size < len(word_list):
        word_list = word_list[0:args.max_vocab_size]
    if args.add_special_symbols:
        print('Add special symbols')
        word_list = ['<S>', '</S>', '<UNK>'] + word_list

    print('Final vocab size: %s' % len(word_list))

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for w in word_list:
            f.write('%s\n' % w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect vocabulary file')
    parser.add_argument('--input_files', help='Input file patterns')
    parser.add_argument('--max_vocab_size', type=int, help='Max vocab size')
    parser.add_argument('--freq_cutoff', type=int, default=1,
        help='Frequency cutoff, only save words with frequency larger than this value')
    parser.add_argument('--output_file', help='Output vocab file')
    parser.add_argument('--add_special_symbols', action='store_true',
        help='If set, will add </S> <S> <UNK> into the vocabulary')

    args = parser.parse_args()
    main(args)
