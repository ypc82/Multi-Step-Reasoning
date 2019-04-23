import os
from multi_corpus import MultiCorpus
import copy
import logging
import pickle
from tqdm import tqdm
import json

def is_pos(corpus, pid, logger=None):
    para = corpus.paragraphs.get(pid, None)
    if para is not None and para.ans_occurance is 0:
        del corpus.paragraphs[pid]
        return False
    return True


"""
This function is to be used as a in memory filter. As input it takes
the loaded pickle file (MultiCorpus object). And filters out all the
negative examples.

By default it deep copies the MultiCorpus object. This can be used
to debug outputs. However, this step will take considerable time.

This function traverses over all the questions in MultiCorpus
object. It finds the paragraphs which have ans_occurance value set
to 0 and deletes them from the question's pids list. Also, it 
deletes the said pid from MultiCorpus object's paragraphs dict.
"""
def label_filter(args, corpus_orig, logger=None, deep_copy=False, save=False, fileName=None):
    corpus = corpus_orig
    if deep_copy:
    	corpus = copy.deepcopy(corpus)
    
    for k, v in tqdm(corpus.questions.items()):
        v.pids = [x for x in v.pids if is_pos(corpus, x, logger)]

    if save:
        filePath = os.path.join(args.data_dir, args.src, "data", args.domain, fileName)
        try:
            with open(filePath + '_no_neg.pkl', 'wb') as output:
                pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)
        except PermissionError:
            logger.error("Cannot dump file in directory, check permissions. No write permission.")
    return corpus

def overlap_filter(overlap_file, data_type):
    scitail_dir = '/mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data/scitail/data/web-open'
    scitail_path = os.path.join(scitail_dir, f'processed_{data_type}.pkl')
    with open(scitail_path, 'rb') as infile:
        scitail_data = pickle.load(infile)

    qid2text_path = os.path.join(scitail_dir, f'qid2text_{data_type}.json')
    with open(qid2text_path) as infile:
        qid2text = json.load(infile)
    
    pid2text_path = os.path.join(scitail_dir, f'pid2text_{data_type}.json')
    with open(pid2text_path) as infile:
        pid2text = json.load(infile)

    with open(overlap_file) as infile:
        overlap_dic = json.load(infile)
    overlap = [lst[1] for lst in overlap_dic.values()]

    overlap_qids = []
    for qid, ques in qid2text.items():
        if ques in overlap:
            overlap_qids.append(qid)

    overlap_pids = []
    for pid, para in scitail_data.paragraphs.items():
        if para.qid in overlap_qids:
            overlap_pids.append(pid)
    
    print('Origin questions:', len(scitail_data.questions))
    for qid in overlap_qids:
        del scitail_data.questions[qid]
    print('Filtered questions:', len(scitail_data.questions))

    print('Origin paragraphs:', len(scitail_data.paragraphs))
    for pid in overlap_pids:
        del scitail_data.paragraphs[pid]
    print('Filtered paragraphs:', len(scitail_data.paragraphs))
    
    print(f'Save file to {scitail_path}')
    with open(scitail_path, 'wb') as f:
        pickle.dump(scitail_data, f)

def remove_overlap():
    overlap_file = 'paragraph_encoder/overlap_C.json'
    #overlap_filter(overlap_file, 'train')
    #overlap_filter(overlap_file, 'dev')
    #overlap_filter(overlap_file, 'test')

if __name__ == '__main__':
    remove_overlap()

