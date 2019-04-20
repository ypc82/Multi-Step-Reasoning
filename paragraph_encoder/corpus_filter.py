import os
from multi_corpus import MultiCorpus
import copy
import logging
import pickle
from tqdm import tqdm


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
