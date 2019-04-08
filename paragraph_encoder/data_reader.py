import os
import sys
import logging
import argparse
import json
import pickle

import spacy
from spacy.lang.en import English
from tqdm import tqdm

from multi_corpus import MultiCorpus

logger = logging.getLogger()
nlp = English()
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]


class ARCReader():
    def __init__(self, args, qa_path, corpus_path):
        self.args = args
        self.qa_path = qa_path
        self.arcCorpus_path = corpus_path

    def read_qafile(self):
        qid, question, choices, answerKey = [], [], [], []
        logger.info(f"Load {self.qa_path}")
        with open(self.qa_path) as infile:
            for line in infile:
                qa_json = json.loads(line)
                qid.append(qa_json["id"])
                question.append(qa_json["question"]["stem"])
                choices.append(qa_json["question"]["choices"])
                answerKey.append(qa_json["answerKey"])
                
        return qid, question, choices, answerKey
                

    def read_corpus(self):
        logger.info(f"Load {self.arcCorpus_path}")
        with open(self.arcCorpus_path) as infile:
            arc_corpus = [line.strip() for line in infile if len(line.strip()) > 0]
        logger.info(f'len(arc_corpus)={len(arc_corpus)}')

        sentences = []
        for doc in tqdm(nlp.pipe(arc_corpus, batch_size=10000, n_threads=32), total=len(arc_corpus)):
            sentences.append([token.text for token in doc])

        return sentences

    def run(self):
        qid, question, choices, answerKey = self.read_qafile()
        logger.info(f'Number of questions={len(qid)}')
        logger.info('Tokenize question')
        tokened_question = []
        for doc in nlp.pipe(question, batch_size=100, n_threads=32):
            tokened_question.append([token.text for token in doc])


        corpus = MultiCorpus(self.args)

        logger.info('Add corpus.questions')
        #import gnureadline
        #import ipdb;ipdb.set_trace()
        for i in tqdm(range(len(qid))):
            qtext = tokened_question[i]
            for choice in choices[i]:
                cqid = qid[i] + '_' + choice['label']
                label = 1 if choice['label'] == answerKey[i] else 0
                corpus.questions[cqid] = corpus.Question(self.args, cqid, qtext, 
                                                         choice_text=tokenize(choice['text']),
                                                         label=label)

        arc_corpus = self.read_corpus()

        logger.info('Add corpus.paragraphs')
        for i in tqdm(range(len(arc_corpus))):
            pid = 'p' + str(i)
            corpus.paragraphs[pid] = corpus.Paragraph(self.args, pid, arc_corpus[i])

        return corpus

class ScitailReader():
    def __init__(self, args, data_path, qid2text_path, pid2text_path):
        self.args = args
        self.data_path = data_path
        self.qid2text_path = qid2text_path
        self.pid2text_path = pid2text_path

    def read_file(self, data_path):
        question, answer, premise, hypothesis, label = [], [], [], [], []
        label2idx = {'entails': 1, 'neutral': 0}

        logger.info(f"Load {data_path}")
        with open(data_path) as infile:
            for line in infile:
                data = json.loads(line)
                question.append(data["question"])
                answer.append(data["answer"])
                premise.append(data["sentence1"])
                hypothesis.append(data["sentence2"])
                label.append(label2idx[data["gold_label"]])
        logger.info(f"Number of data: {len(question)}")

        return question, answer, premise, hypothesis, label

    def indexify(self, text_list, prefix, file_path):
        idx_list = []
        if os.path.isfile(file_path):
            logger.info(f'Load {file_path}')
            idx2text = json.load(open(file_path))
            text2idx = {v: k for k, v in idx2text.items()}
            for text in text_list:
                idx_list.append(text2idx[text])
        else:
            text2idx = {}
            for text in text_list:
                # Convert to index
                if text not in text2idx:
                    text2idx[text] = prefix+str(len(text2idx))
                idx_list.append(text2idx[text])
            idx2text = {v: k for k, v in text2idx.items()}
            logger.info(f'Save to {file_path}')
            with open(file_path, 'w') as outfile:
                json.dump(idx2text, outfile, indent=4)

        return idx_list, idx2text

    def run(self):
        # MultiCorpus: args, tfidf, questions, paragraphs
        corpus = MultiCorpus(self.args)

        # Read file
        question, answer, premise, hypothesis, label = self.read_file(self.data_path)

        # Indexify
        question, qid2text = self.indexify(question, 'q', self.qid2text_path)
        premise, pid2text = self.indexify(premise, 'p', self.pid2text_path)
        logger.info(f'len(qid2text)={len(qid2text)}')
        logger.info(f'len(pid2text)={len(pid2text)}')

        # Add Question and Paragraph to MultiCorpus object
        for i in tqdm(range(len(question))):
            cqid = question[i]
            cpid = premise[i] + '_' + cqid
            if cqid in corpus.questions:
                corpus.questions[cqid].pids.append(cpid)
            else:
                qtext = tokenize(qid2text[cqid])
                corpus.questions[cqid] = corpus.Question(self.args, cqid, qtext, [cpid])
            
            ptext = tokenize(pid2text[premise[i]])
            corpus.paragraphs[cpid] = corpus.Paragraph(
                                        self.args, cpid, ptext, cqid, 
                                        ans_occurance=label[i], 
                                        ans=tokenize(answer[i]), 
                                        reform_qtext=tokenize(hypothesis[i])
                                      )

        return corpus

def read(dataset, args, data_type='train'):
    data_dir = '/mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data'

    if dataset == 'arc_dgem':
        pass
        arc_dir = '/home/yipeichen/ARC-Solvers/data/ARC-V1-Feb2018'
        qa_file = os.path.join(arc_dir, 'ARC-Challenge/ARC-Challenge-Train_dgem_onlyAns_filtered.json')
        corpus_file = os.path.join(arc_dir, 'ARC_Corpus.txt')
        outfile_path = os.path.join(data_dir, 'arc_dgem/data/web-open/processed_train.pkl')


    elif dataset == 'arc_corpus':
        arc_dir = '/home/yipeichen/ARC-Solvers/data/ARC-V1-Feb2018'
        qa_path = os.path.join(arc_dir, f'ARC-Challenge/ARC-Challenge-{data_type}.jsonl')
        corpus_path = os.path.join(arc_dir, 'ARC_Corpus_1K.txt')
        output_path = os.path.join(data_dir, 'arc/data/web-open/processed_test_1K.pkl')

        reader = ARCReader(args, qa_path, corpus_path)

    elif dataset == 'scitail':
        scitail_dir = '/mnt/nfs/work1/mccallum/yipeichen/data/SciTailV1.1/predictor_format'
        filename = f'scitail_1.0_structure_{data_type}.jsonl'
        data_path = os.path.join(scitail_dir, filename)

        qid2text_path = os.path.join(data_dir, dataset, 'data/web-open', f'qid2text_{data_type}.json')
        pid2text_path = os.path.join(data_dir, dataset, 'data/web-open', f'pid2text_{data_type}.json')
        output_path = os.path.join(data_dir, dataset, 'data/web-open', f'processed_{data_type}.pkl')

        reader = ScitailReader(args, data_path, qid2text_path, pid2text_path)

    corpus = reader.run()

    logger.info(f'Save to {output_path}')
    with open(output_path, 'wb') as outfile:
        pickle.dump(corpus, outfile)


if __name__ == "__main__":
    # Set logging
    logger.setLevel(logging.INFO)
    fomatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fomatter)
    logger.addHandler(console)
    logger.info('[COMMAND] %s' % ' '.join(sys.argv))

    multicorpus_parser = argparse.ArgumentParser()
    multicorpus_parser.add_argument("--calculate_tfidf", default=False, action='store_true') 
    multicorpus_parser.add_argument("--small", default=False, action='store_true') 
    options = multicorpus_parser.parse_args()

    #read('scitail', options, 'train')
    #read('scitail', options, 'dev')
    #read('scitail', options, 'test')

    read('arc_corpus', options, 'Test')

