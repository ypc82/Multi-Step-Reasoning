#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
# Few methods have been adapted from https://github.com/facebookresearch/DrQA
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data processing/loading helpers."""


import numpy as np
import json
import logging
from smart_open import smart_open
import unicodedata
import heapq
import os

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize

import arc_random_sampler
from arc_random_sampler import EsSearch

logger = logging.getLogger()

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL) # or desired level
tracer.addHandler(logging.FileHandler('indexer.log'))

# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self, args):
        self.args = args
        if not args.create_vocab:
            logger.info('[ Reading vocab files from {}]'.format(args.vocab_dir))
            self.tok2ind = json.load(open(args.vocab_dir+'tok2ind.json'))
            self.ind2tok = json.load(open(args.vocab_dir+'ind2tok.json'))

        else:
            self.tok2ind = {self.NULL: 0, self.UNK: 1}
            self.ind2tok = {0: self.NULL, 1: self.UNK}
            self.oov_words = {}

            # Index words in embedding file
            if args.pretrained_words and args.embedding_file:
                logger.info('[ Indexing words in embedding file... ]')
                self.valid_words = set()
                with smart_open(args.embedding_file) as f:
                    for line in f:
                        w = self.normalize(line.decode('utf-8').rstrip().split(' ')[0])
                        self.valid_words.add(w)
                logger.info('[ Num words in set = %d ]' % len(self.valid_words))
            else:
                self.valid_words = None

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def add(self, token):
        token = self.normalize(token)
        if self.valid_words and token not in self.valid_words:
            # logger.info('{} not a valid word'.format(token))
            if token not in self.oov_words:
                self.oov_words[token] = len(self.oov_words)
            return
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def swap_top(self, top_words):
        """
        Reindexes the dictionary to have top_words labelled 2:N.
        (0, 1 are for <NULL>, <UNK>)
        """
        for idx, w in enumerate(top_words, 2):
            if w in self.tok2ind:
                w_2, idx_2 = self.ind2tok[idx], self.tok2ind[w]
                self.tok2ind[w], self.ind2tok[idx] = idx, w
                self.tok2ind[w_2], self.ind2tok[idx_2] = idx_2, w_2

    def save(self):

        fout = open(os.path.join(self.args.vocab_dir, "ind2tok.json"), "w")
        json.dump(self.ind2tok, fout)
        fout.close()
        fout = open(os.path.join(self.args.vocab_dir, "tok2ind.json"), "w")
        json.dump(self.tok2ind, fout)
        fout.close()
        logger.info("Dictionary saved at {}".format(self.args.vocab_dir))


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class SquadDataset(Dataset):
    def __init__(self, args, examples, word_dict,
                 feature_dict, single_answer=False, para_mode=False, train_time=True):
        self.examples = examples
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.args = args
        self.single_answer = single_answer
        self.para_mode = para_mode
        self.train_time = train_time

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.args, self.examples[index], self.word_dict, self.feature_dict, self.single_answer,
                         self.para_mode, self.train_time)

    def lengths(self):
        if not self.para_mode:
            return [(len(ex['document']), len(ex['question'])) for ex in self.examples]
        else:
            q_key = 'question_str' if (self.args.src == 'triviaqa' or self.args.src == 'qangaroo') else 'question'
            return [(len(ex['document']), max([len(para) for para in ex['document']]), len(ex[q_key])) for ex in self.examples]

class MultiCorpusDataset(Dataset):
    def __init__(self, args, corpus, word_dict,
                 feature_dict, single_answer=False, para_mode=True, train_time=True):
        self.corpus = corpus
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.args = args
        self.single_answer = single_answer
        self.para_mode = para_mode
        self.train_time = train_time
        self.pid_list = list(self.corpus.paragraphs.keys())
        self.qid_list = list(self.corpus.questions.keys())
        self.total_para_num = len(self.corpus.paragraphs)
        #if self.train_time and self.args.augment_train:
        if self.args.augment_train:
            # TODO: set client=node008
            es_search = EsSearch(es_client="node008")
            self.add_para_list = self.sample_negative_ex(es_search)
            logger.info(f'Add {len(self.add_para_list)} negative samples from {self.args.augment_dataset}')

    def __len__(self):
        if self.para_mode:
            #if self.train_time and self.args.augment_train:
            if self.args.augment_train:
                return len(self.corpus.paragraphs) + len(self.add_para_list)
            return len(self.corpus.paragraphs)
        else:
            return len(self.corpus.questions)

    def __getitem__(self, index):
        if self.para_mode:
            if index < len(self.pid_list):
                ex = self.get_ex(index)
                return vectorize(self.args, ex)
            else:
                ex = self.get_additional_ex(index)
                return vectorize(self.args, ex)

        else:
            raise NotImplementedError("later")

    def get_ex(self, index):
        ex = {}
        pid =  self.pid_list[index]
        para = self.corpus.paragraphs[pid]
        assert pid == para.pid
        ex['document'] = para.text
        ex['id'] = para.pid
        ex['ans_occurance'] = para.ans_occurance
        
        if self.args.src == 'scitail' and self.args.experiment_name == 'hypothesis':
            ex['question'] = para.reform_qtext

        elif self.args.src == 'scitail' and self.args.experiment_name == 'question_answer':
            qid = para.qid
            question = self.corpus.questions[qid]
            ex['question'] = question.text + para.ans

        else:
            qid = para.qid
            question = self.corpus.questions[qid]
            ex['question'] = question.text
            assert pid in question.pids

        return ex

    def get_additional_ex(self, index):
        ex = {}
        idx = index-len(self.pid_list)
        para_dic =  self.add_para_list[idx]

        ex['ans_occurance'] = 0
        ex['document'] = para_dic["paragraph"]
        ex['id'] = f'add_{para_dic["qid"]}_{str(idx)}'
        ex['question'] = para_dic["question"]

        # TODO: args.augment_dataset == 'scitail'

        return ex

    def sample_negative_ex(self, es_search):
        # add_para_list is a list of dictionaries, each of which contains keys:
        # "paragraph", "question", "qid" 
        para_list = []
        
        # Get 20K Elastic Search result at a time
        es_result = []
        while len(es_result) < 20000:
            es_result.extend(es_search.get_hits(max_hits_retrieved=10000, max_filtered_hits=10000,
                                     max_hit_length=self.args.max_hit_len, min_hit_length=self.args.min_hit_len,
                                     random_seed=len(es_result)))
        print(f'ES result: {len(es_result)}')

        for idx, qid in enumerate(self.qid_list):
            q_obj = self.corpus.questions[qid]

            # obtain origin pos:neg
            neg = 0
            pids = q_obj.pids
            for pid in pids:
                if self.corpus.paragraphs[pid].ans_occurance == 0:
                    neg += 1
            pos = len(pids) - neg

            add_counter = 0
            try:
                while neg/pos < self.args.neg_sample_rate:
                    ex_dic = {}
                    ex_dic["qid"] = qid
                    ex_dic["question"] = q_obj.text
                    ex_dic["paragraph"] = es_result[len(para_list)].text
                    para_list.append(ex_dic)
                    neg += 1
            except:
                print(f'[{idx}] neg/pos: {neg/pos:.3f} ({neg}/{pos}); add_counter: {len(para_list)}/{len(es_result)}')
        return para_list


class ArcDataset(Dataset):
    def __init__(self, args, data_dic, data_type):
        self.args = args
        self.data_dic = data_dic
        self.data_type = data_type
        self.word_dict = args.word_dict
        self.data_list = list(data_dic.keys())

    def __len__(self):
        return len(self.data_dic)

    def __getitem__(self, index):
        ex_id =  self.data_list[index]
        if self.data_type == 'para':
            para = self.data_dic[ex_id]
            words = para.text

        elif self.data_type == 'ques':
            qtext = self.data_dic[ex_id].text
            choice = self.data_dic[ex_id].choice_text
            words = qtext + choice

        words = [self.word_dict[w] for w in words]

        return words, ex_id

# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, para_mode=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.para_mode = para_mode

    def __iter__(self):
        if not self.para_mode:
            lengths = np.array(
                [(-l[0], -l[1], np.random.random()) for l in self.lengths],
                dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
            )
        else:
            lengths = np.array([(-l[0], -l[1], -l[2], np.random.random()) for l in self.lengths], dtype=[('l1', np.int_), ('l2', np.int_), ('l3', np.int_), ('rand', np.float_)])
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand')) if not self.para_mode else np.argsort(lengths, order=('l1', 'l2', 'l3', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

class CorrectParaSortedBatchSampler(Sampler):
    """
    This awesome sampler was written by Peng Qi (http://qipeng.me/)
    """
    def __init__(self, dataset, batch_size, shuffle=True, para_mode=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.para_mode = para_mode

    def __iter__(self):
        import sys
        correct_paras = [(ex[5] > 0).long().sum() for ex in self.dataset]

        # make sure the number of correct paras in each minibatch is about the same
        mean = sum(correct_paras) / len(correct_paras)
        target = mean * self.batch_size

        # also make sure the number of total paras in each minibatch is about the same
        lengths = [x[0] for x in self.dataset.lengths()]
        target2 = sum(lengths) / len(lengths) * self.batch_size

        heuristic_weight = 0.1 # heuristic importance of making sum_para_len uniform compared to making sum_correct_paras uniform

        indices = [x[0] for x in sorted(enumerate(zip(correct_paras, lengths)), key=lambda x: x[1], reverse=True)]

        batches = [[] for _ in range((len(self.dataset) + self.batch_size - 1) // self.batch_size)]

        batches_by_size = {0: {0: [(i, 0, 0) for i in range(len(batches))] } }

        K = 100 # "beam" size

        for idx in indices:
            costs = []
            for size in batches_by_size:
                cost_reduction = -(2 * size + correct_paras[idx] - 2 * target) * correct_paras[idx]

                costs += [(size, cost_reduction)]

            costs = heapq.nlargest(K, costs, key=lambda x: x[1])

            best_cand = None
            for size, cost in costs:
                best_size2 = -1
                best_reduction = -float('inf')
                for size2 in batches_by_size[size]:
                    cost_reduction = -(2 * size2 + lengths[idx] - 2 * target2) * lengths[idx]

                    if cost_reduction > best_reduction:
                        best_size2 = size2
                        best_reduction = cost_reduction

                assert best_size2 >= 0

                cost_reduction_all = cost + best_reduction * heuristic_weight
                if best_cand is None or cost_reduction_all > best_cand[2]:
                    best_cand = (size, best_size2, cost_reduction_all, cost, best_reduction)

            assert best_cand is not None

            best_size, best_size2 = best_cand[:2]

            assert len(batches_by_size[best_size]) > 0

            # all batches of the same size are created equal
            best_batch, batches_by_size[best_size][best_size2] = batches_by_size[best_size][best_size2][0], batches_by_size[best_size][best_size2][1:]

            if len(batches_by_size[best_size][best_size2]) == 0:
                del batches_by_size[best_size][best_size2]
                if len(batches_by_size[best_size]) == 0:
                    del batches_by_size[best_size]

            batches[best_batch[0]] += [idx]
            newsize = best_batch[1] + correct_paras[idx]
            newsize2 = best_batch[2] + lengths[idx]

            if len(batches[best_batch[0]]) < self.batch_size:
                # insert back
                if newsize not in batches_by_size:
                    batches_by_size[newsize] = {}

                if newsize2 not in batches_by_size[newsize]:
                    batches_by_size[newsize][newsize2] = [(best_batch[0], newsize, newsize2)]
                else:
                    batches_by_size[newsize][newsize2] += [(best_batch[0], newsize, newsize2)]

        if self.shuffle:
            np.random.shuffle(batches)

        return iter([x for batch in batches for x in batch])

    def __len__(self):
        return len(self.dataset)
