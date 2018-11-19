#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:12 PM
import random
import math
import numpy as np
import ujson as json


feature_list = ['qem', 'pem', 'fx', 'fx2', 'fx3',
                'qex1', 'qex2', 'qex3', 'qex4', 'qex5', 'qex6', 'qex7', 'qex8', 'qex9']


class SQuAD(object):
    def __init__(self, args):

        self.max_context_len = args.max_context_len
        self.max_question_len = args.max_question_len
        self.max_word_len = args.max_word_len
        self.max_answer_len = args.max_answer_len

        self.train_samples = self.load_samples(args.train_file,
                                               drop_invalid=True) if args.train_file is not None else []
        self.dev_samples = self.load_samples(args.dev_file) if args.dev_file is not None else []
        self.test_samples = self.load_samples(args.test_file) if args.test_file is not None else []
        self.augment_dict = self.build_augment_dict(args.augment_files, drop_invalid=True) \
            if args.augment_files is not None and args.augment_data_ratio > 0 else None
        self.feat_dict = self.build_feature_dict(args.feature_files) \
            if args.feature_files is not None and 'feature' in args.model else None

        self.word_vocab = None
        self.char_vocab = None

    def build_augment_dict(self, file_list, drop_invalid=False):
        augment_dict = {}
        for file in file_list:
            for sample in self.load_samples(file, drop_invalid=drop_invalid):
                s_id = sample['id'].replace('zh-CN', '').replace('it', '')
                if s_id in augment_dict:
                    augment_dict[s_id].append(sample)
                else:
                    augment_dict[s_id] = [sample]
        return augment_dict

    def build_feature_dict(self, file_list):
        feat_dict = {}
        for file in file_list:
            fin = open(file, 'r')
            json_obj = json.load(fin)
            data = json_obj if isinstance(json_obj, list) else json_obj['data']
            fin.close()
            for sample in data:
                assert sample['uid'] not in feat_dict, 'Duplicate features for question {}'.format(sample['uid'])
                p_len, q_len = sample['plen'], sample['qlen']
                p_feat_vec, q_feat_vec = [], []
                for feat in feature_list:
                    feat_values = sample[feat]
                    if len(feat_values) == p_len:
                        p_feat_vec.append(feat_values)
                        q_feat_vec.append([0] * q_len)
                    elif len(feat_values) == q_len:
                        q_feat_vec.append(feat_values)
                        p_feat_vec.append([0] * p_len)
                    else:
                        raise ValueError
                p_feat_vec = np.transpose(np.array(p_feat_vec)).tolist()
                q_feat_vec = np.transpose(np.array(q_feat_vec)).tolist()
                feat_dict[sample['uid']] = {'uid': sample['uid'], 'plen': p_len, 'qlen': q_len,
                                            'p_feats': p_feat_vec, 'q_feats': q_feat_vec}
        return feat_dict

    def load_samples(self, file, drop_invalid=False):
        samples = []
        with open(file, 'r') as fin:
            json_obj = json.load(fin)
            for article in json_obj['data']:
                for para in article['paragraphs']:
                    context = para['context']
                    context_tokens = para['tokenized_context']
                    context_words = [token['text'] for token in context_tokens]
                    context_word_offsets = [(token['beginPosition'], token['beginPosition'] + token['length'] - 1)
                                            for token in context_tokens]
                    for qa in para['qas']:
                        question = qa['question']
                        question_id = qa['id']
                        question_tokens = qa['tokenized_question']
                        question_words = [token['text'] for token in question_tokens]
                        answer_spans = []
                        answer_texts = []
                        for answer in qa['answers']:
                            if drop_invalid and answer['is_token_mismatch']:
                                continue
                            answer_text = answer['text']
                            answer_texts.append(answer_text)
                            answer_spans.append((answer['answer_token_start'],
                                                 answer['answer_token_start'] + answer['answer_token_count'] - 1))
                        sample = {'context': context, 'question': question,
                                  'context_words': context_words, 'question_words': question_words,
                                  'context_word_offsets': context_word_offsets,
                                  'answer_spans': answer_spans, 'answers': answer_texts, 'id': question_id}
                        if drop_invalid and len(sample['answer_spans']) == 0:
                            continue
                        if drop_invalid and sample['answer_spans'][0][1] >= self.max_context_len:
                            continue
                        # if len(sample['context_words']) > self.max_context_len and drop_invalid:
                        #     continue
                        samples.append(sample)
                        # if len(samples) >= 1000:
                        #     return samples
        return samples

    def gen_words(self, train=False, dev=False, test=False):
        samples = []
        if train:
            samples += self.train_samples
        if dev:
            samples += self.dev_samples
        if test:
            samples += self.test_samples
        for sample in samples:
            for word in sample['context_words']:
                yield word
            for word in sample['question_words']:
                yield word

    def gen_mini_batches(self, batch_size, train=False, dev=False, test=False,
                         shuffle=False, bucket_every_n_batch=20, augment_ratio=0.0, word_drop_ratio=0.0):
        assert not (augment_ratio > 0 and self.augment_dict is None), 'No augmented data is loaded!'
        samples = []
        if train:
            samples += self.train_samples
        if dev:
            samples += self.dev_samples
        if test:
            samples += self.test_samples

        if train and augment_ratio > 0:
            augment_samples = [it2 for it in self.augment_dict.values() for it2 in it]
            random.shuffle(augment_samples)
            samples += augment_samples[: int(len(samples) * augment_ratio)]

        data_size = len(samples)
        batch_num = math.ceil(1.0 * data_size / batch_size)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        if bucket_every_n_batch > 1:
            bucket_num = int(math.ceil(batch_num / bucket_every_n_batch))
            for bucket_idx in range(bucket_num):
                bucket_indices = indices[batch_size * bucket_every_n_batch * bucket_idx:
                                         batch_size * bucket_every_n_batch * (bucket_idx + 1)]
                sorted_bucket_indices = sorted(bucket_indices,
                                               key=lambda idx: len(samples[idx]['context_words']), reverse=True)
                for batch_start in range(0, len(sorted_bucket_indices), batch_size):
                    batch_indices = sorted_bucket_indices[batch_start: batch_start + batch_size]
                    yield self.one_mini_batch(samples, batch_indices, word_drop_ratio)
        else:
            for batch_start in np.arange(0, data_size, batch_size):
                batch_indices = indices[batch_start: batch_start + batch_size]
                yield self.one_mini_batch(samples, batch_indices, word_drop_ratio)

    def one_mini_batch(self, samples, batch_indices, word_drop_ratio=0.0):
        raw_data = [samples[i] for i in batch_indices]

        # if augment_ratio > 0:
        #     for idx in range(len(raw_data)):
        #         question_id = raw_data[idx]['id']
        #         if question_id in augment_data_dict:
        #             augment_samples = augment_data_dict[question_id]
        #             if random.random() < augment_ratio and len(augment_samples) > 0:
        #                 raw_data[idx] = random.choice(augment_samples)

        batch_data = {'raw_data': raw_data,
                      'context_word_ids': [], 'question_word_ids': [],
                      'context_char_ids': [], 'question_char_ids': [],
                      'start_pos': [], 'end_pos': []}
        if self.feat_dict is not None:
            batch_data['context_feats'] = [self.feat_dict[sample['id']]['p_feats'] for sample in raw_data]
            batch_data['question_feats'] = [self.feat_dict[sample['id']]['q_feats'] for sample in raw_data]

        for sidx, sample in enumerate(batch_data['raw_data']):
            context_word_ids = self.word_vocab.convert_to_ids(
                self.drop_tokens_as_unk(sample['context_words'], word_drop_ratio, self.word_vocab.unk_token)
                if word_drop_ratio > 0 else sample['context_words']
            )
            question_word_ids = self.word_vocab.convert_to_ids(
                # self.drop_tokens_as_unk(sample['question_words'], word_drop_ratio, self.word_vocab.unk_token)
                # if word_drop_ratio > 0 else sample['question_words']
                sample['question_words']
            )
            context_char_ids = [self.char_vocab.convert_to_ids(list(word)) for word in sample['context_words']]
            # context_char_ids = [self.char_vocab.convert_to_ids(
            #     self.drop_tokens_as_unk(list(word), word_drop_ratio, self.char_vocab.unk_token)
            #     if word_drop_ratio > 0 else list(word))
            #     for word in sample['context_words']]
            question_char_ids = [self.char_vocab.convert_to_ids(list(word)) for word in sample['question_words']]
            # question_char_ids = [self.char_vocab.convert_to_ids(
            #     self.drop_tokens_as_unk(list(word), word_drop_ratio, self.char_vocab.unk_token)
            #     if word_drop_ratio > 0 else list(word))
            #     for word in sample['question_words']]
            batch_data['context_word_ids'].append(context_word_ids)
            batch_data['question_word_ids'].append(question_word_ids)
            batch_data['context_char_ids'].append(context_char_ids)
            batch_data['question_char_ids'].append(question_char_ids)
            batch_data['start_pos'].append(sample['answer_spans'][0][0])
            batch_data['end_pos'].append(sample['answer_spans'][0][1])
        return self.pad_and_cut(batch_data, 0, 0)

    @staticmethod
    def drop_tokens_as_unk(tokens, drop_rate, unk_token, binomial=True):
        new_tokens = tokens[:]
        if binomial:
            drop_num = int(np.random.binomial(len(new_tokens), drop_rate))
        else:
            drop_num = round(len(new_tokens) * drop_rate)
        ids_to_drop = random.sample(range(len(new_tokens)), drop_num)
        for id in ids_to_drop:
            new_tokens[id] = unk_token
        return new_tokens

    def pad_and_cut(self, batch_data, word_pad_id, char_pad_id):
        max_context_len = min(self.max_context_len, max([len(word_ids) for word_ids in batch_data['context_word_ids']]))
        max_question_len = min(self.max_question_len, max([len(word_ids) for word_ids in batch_data['question_word_ids']]))
        batch_data['context_word_ids'] = [(ids + [word_pad_id] * (max_context_len - len(ids)))[:max_context_len]
                                          for ids in batch_data['context_word_ids']]
        batch_data['question_word_ids'] = [(ids + [word_pad_id] * (max_question_len - len(ids)))[:max_question_len]
                                           for ids in batch_data['question_word_ids']]
        batch_data['context_char_ids'] = [(context_char_ids + [[]] * (max_context_len - len(context_char_ids)))[:max_context_len]
                                          for context_char_ids in batch_data['context_char_ids']]
        batch_data['question_char_ids'] = [(question_char_ids + [[]] * (max_question_len - len(question_char_ids)))[:max_question_len]
                                           for question_char_ids in batch_data['question_char_ids']]
        max_context_char_len = min(self.max_word_len,
                                   max([len(char_ids) for context_char_ids in batch_data['context_char_ids']
                                        for char_ids in context_char_ids]))
        max_question_char_len = min(self.max_word_len,
                                    max([len(char_ids) for question_char_ids in batch_data['question_char_ids']
                                         for char_ids in question_char_ids]))
        batch_data['context_char_ids'] = [
            [(ids + [char_pad_id] * (max_context_char_len - len(ids)))[:max_context_char_len]
             for ids in context_char_ids] for context_char_ids in batch_data['context_char_ids']]
        batch_data['question_char_ids'] = [
            [(ids + [char_pad_id] * (max_question_char_len - len(ids)))[:max_question_char_len]
             for ids in question_char_ids] for question_char_ids in batch_data['question_char_ids']]
        if 'context_feats' in batch_data:
            batch_data['context_feats'] = [
                (feats + [[0.0] * len(feats[0])] * (max_context_len - len(feats)))[:max_context_len]
                for feats in batch_data['context_feats']]
        if 'question_feats' in batch_data:
            batch_data['question_feats'] = [
                (feats + [[0.0] * len(feats[0])] * (max_question_len - len(feats)))[:max_question_len]
                for feats in batch_data['question_feats']]
        return batch_data


class SQuAD2(SQuAD):
    def load_samples(self, file, drop_invalid=False):
        samples = []
        with open(file, 'r') as fin:
            json_obj = json.load(fin)
            for article in json_obj['data']:
                for para in article['paragraphs']:
                    context = para['context']
                    context_tokens = para['tokenized_context']
                    context_words = [token['text'] for token in context_tokens[: self.max_context_len - 1]] + ['<nan>']
                    context_word_offsets = [(token['beginPosition'], token['beginPosition'] + token['length'] - 1)
                                            for token in context_tokens[: self.max_context_len - 1]]
                    for qa in para['qas']:
                        question = qa['question']
                        question_id = qa['id']
                        question_tokens = qa['tokenized_question']
                        question_words = [token['text'] for token in question_tokens]
                        answer_spans = []
                        answer_texts = []
                        if not qa['is_impossible'] and len(qa['answers']) > 0:
                            for answer in qa['answers']:
                                if drop_invalid and answer['is_token_mismatch']:
                                    continue
                                answer_text = answer['text']
                                answer_texts.append(answer_text)
                                answer_spans.append((answer['answer_token_start'],
                                                     answer['answer_token_start'] + answer['answer_token_count'] - 1))
                        else:
                            answer_texts.append('')
                            answer_spans.append((len(context_words) - 1, len(context_words) - 1))
                        sample = {'context': context, 'question': question,
                                  'context_words': context_words, 'question_words': question_words,
                                  'context_word_offsets': context_word_offsets,
                                  'answer_spans': answer_spans, 'answers': answer_texts, 'id': question_id}
                        if drop_invalid and len(sample['answer_spans']) == 0:
                            continue
                        if drop_invalid and sample['answer_spans'][0][1] >= self.max_context_len:
                            continue
                        samples.append(sample)
                        # if 'train' in file and len(samples) >= 100:
                        #     return samples
        return samples

