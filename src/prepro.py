#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:10 PM
import os
import pickle
import spacy
import ujson as json
from tqdm import tqdm
from dataset import SQuAD
from vocab import Vocab

spacy_nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'vectors', 'textcat'])


def word_tokenize(sent):
    results = []
    for token in spacy_nlp(sent):
        hyphen_found = None
        for hyphen in ['-', '–', '~']:
            if hyphen in token.text:
                hyphen_found = hyphen
                break
        if hyphen_found is not None:
            char_offset = token.idx
            for sub_str in token.text.split(hyphen_found):
                if sub_str:
                    results.append({"text": sub_str, "beginPosition": char_offset,
                                    "length": len(sub_str), "index": len(results)})
                    char_offset += len(sub_str)
                results.append({"text": hyphen_found, "beginPosition": char_offset,
                                "length": len(hyphen_found), "index": len(results)})
                char_offset += len(hyphen_found)
            results.pop(-1)
        else:
            results.append({"text": token.text, "beginPosition": token.idx,
                            "length": len(token), "index": len(results)})

    return results


def extract_and_format_samples(filename, data_type):
    print('Processing {} samples...'.format(data_type))
    total = 0
    mismatch_cnt = 0
    with open(filename, 'r') as fh:
        data = json.load(fh)
        for article in tqdm(data['data']):
            for para in article['paragraphs']:
                context = para['context'].rstrip()
                context_tokens = word_tokenize(context)
                para['tokenized_context'] = context_tokens
                for qa in para['qas']:
                    total += 1
                    ques = qa['question'].strip().replace("\n", "")
                    question_tokens = word_tokenize(ques)
                    qa['tokenized_question'] = question_tokens
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text) - 1
                        answer_word_ids = []
                        for idx, token in enumerate(context_tokens):
                            char_span = (token['beginPosition'], token['beginPosition'] + token['length'] - 1)
                            if answer_start <= char_span[1] and answer_end >= char_span[0]:
                                answer_word_ids.append(idx)
                        answer['answer_token_start'] = answer_word_ids[0]
                        answer['answer_token_count'] = len(answer_word_ids)
                        char_start = context_tokens[answer_word_ids[0]]['beginPosition']
                        if context[char_start: char_start + len(answer_text)] != answer_text:
                            answer['is_token_mismatch'] = True
                            mismatch_cnt += 1
                        else:
                            answer['is_token_mismatch'] = False
    print("{} mismatch answers".format(mismatch_cnt))
    return data


def save_data_to_json(data, json_file):
    with open(json_file, 'w') as fout:
        fout.write(json.dumps(data) + '\n')


def prepro(args):
    if args.raw_train_file is not None and args.train_file is not None:
        train_data = extract_and_format_samples(args.raw_train_file, 'train')
        save_data_to_json(train_data, args.train_file)

    if args.raw_dev_file is not None and args.dev_file is not None:
        dev_data = extract_and_format_samples(args.raw_dev_file, 'dev')
        save_data_to_json(dev_data, args.dev_file)

    if args.raw_test_file is not None and args.test_file is not None:
        test_data = extract_and_format_samples(args.raw_test_file, 'test')
        save_data_to_json(test_data, args.test_file)


def create_vocab(args):
    print('Creating vocabulary...')
    squad_data = SQuAD(args)
    word_vocab = Vocab(lower=True)
    char_vocab = Vocab(lower=False)
    for word in squad_data.gen_words(train=True, dev=True, test=True):
        if word.strip():
            word_vocab.add(word.strip())
        for char in list(word):
            if char.strip():
                char_vocab.add(char.strip())
    if args.word_embed_path is not None:
        word_vocab.load_pretrained_embeddings(args.word_embed_path)
    else:
        word_vocab.filter_tokens_by_cnt(min_cnt=5)
        word_vocab.embed_dim = args.word_embed_size
    if args.char_embed_path is not None:
        char_vocab.load_pretrained_embeddings(args.char_embed_path)
    else:
        char_vocab.filter_tokens_by_cnt(min_cnt=200)
        char_vocab.embed_dim = args.char_embed_size
    print('{} words and {} chars in the vocabulary'.format(word_vocab.size(), char_vocab.size()))
    print('Saving vocab...')
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
    with open(os.path.join(args.vocab_dir, 'word.vocab'), 'wb') as fout:
        pickle.dump(word_vocab, fout)
    with open(os.path.join(args.vocab_dir, 'char.vocab'), 'wb') as fout:
        pickle.dump(char_vocab, fout)
