import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from pathlib import Path
import argparse
import time

class MultiCorpus:
    class Paragraph:
        def __init__(self, args, pid, text, qid=None, 
                     answer_span=None, ans_occurance=None, 
                     tfidf=None, ans=None, reform_qtext=None,
                     category=None):
            """
            :param args:
            :param pid:
            :param text:
            :param answer_span: numpy array of size num_occ X 2
            :param qid:
            :param tfidf:
            """
            self.args = args
            self.pid = pid # 41141_para_0; 41141_para_4
            self.text = text 
            self.answer_span = answer_span # array([], dtype=float64); [[21 21]]
            self.ans_occurance = ans_occurance # answer_span.shape[0] # 0; 1
            self.qid = qid # 41141; 41141
            self.tfidf_score = tfidf # 0.6386464443605253; 0.6386464443605253
            self.model_score = None # doesn't have this field; doesn't have this field

            # Customize
            self.ans = ans
            self.reform_qtext = reform_qtext
            self.category = category
            

    class Question:
        def __init__(self, args, qid, text, pids=None, choice_text=None, reform_qtext=None, label=None, category=None):
            self.args = args # Namespace(calculate_tfidf=True, small=False)
            self.qid = qid # '140951'
            self.text = text # ['ft', 'sill', ',', 'okla', 'made', 'plea', ',', 'arizona', 'land', ',', 'home', ',', 'father', "'s", 'land', ',', 'ask', 'return']
            self.pids = pids # ['140951_para_0', '140951_para_1', '140951_para_2', '140951_para_3']

            # Customize
            self.choice_text = choice_text
            self.reform_qtext = reform_qtext 
            self.label = label
            self.category = category

    def __init__(self, args):

        self.args = args
        self.tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="english")
        self.questions = {}
        self.paragraphs = {}

    def dists(self, question, paragraphs):

        text = []
        for para in paragraphs:
            text.append(" ".join("".join(s) for s in para.text))
        try:
            para_features = self.tfidf.fit_transform(text)
            q_features = self.tfidf.transform([" ".join(question)])
        except:
            print("tfidf fit_transform threw an exception")
            return [(paragraphs[i], float('inf')) for i in paragraphs]

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.lexsort(([x.start for x in paragraphs], dists))  # in case of ties, use the earlier paragraph
        return [(paragraphs[i], dists[i]) for i in sorted_ix]


    def dists_text(self, question, paragraph_texts):
        """
        modified dist which takes in only paragraph object
        :param question:
        :param paragraphs:
        :return:
        """
        text = []
        for para in paragraph_texts:
            text.append(" ".join(para))

        try:
            para_features = self.tfidf.fit_transform(text)
            q_features = self.tfidf.transform([question])
        except:
            print("tfidf fit_transform threw an exception")
            return [(paragraph_texts[i], float('inf')) for i in paragraph_texts]

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.argsort(dists)
        return [(paragraph_texts[i], dists[i]) for i in sorted_ix]

    def addQuestionParas(self, qid, qtext, paragraphs):

        # for para in paragraphs:
        #     para.text = [w.encode("ascii", errors="ignore").decode() for w in para.text]
        scores = None
        if self.args.calculate_tfidf:
            scores = self.dists(qtext, paragraphs)

        para_ids = []
        for p_counter, p in enumerate(paragraphs):
            tfidf_score = float('inf')
            if scores is not None:
                _, tfidf_score = scores[p_counter]

            pid = qid + "_para_" + str(p_counter)
            para_ids.append(pid)
            paragraph = self.Paragraph(self.args, pid, p.text, p.answer_spans, qid, tfidf_score)
            self.paragraphs[pid] = paragraph

        question = self.Question(self.args, qid, qtext, para_ids)

        self.questions[qid] = question

    
    def addQuestionParas(self, qid, qtext, paragraph_texts, paragraph_answer_spans):

        # for para in paragraphs:
        #     para.text = [w.encode("ascii", errors="ignore").decode() for w in para.text]
        scores = None
        if self.args.calculate_tfidf:
            scores = self.dists_text(" ".join(qtext), paragraph_texts)

        para_ids = []
        for p_counter, p_text in enumerate(paragraph_texts):
            tfidf_score = float('inf')
            if scores is not None:
                _, tfidf_score = scores[p_counter]

            pid = qid + "_para_" + str(p_counter)
            para_ids.append(pid)
            #paragraph = self.Paragraph(self.args, pid, p_text, paragraph_answer_spans[p_counter], qid, tfidf_score)
            paragraph = self.Paragraph(self.args, pid, p_text, None, qid, tfidf_score)
            self.paragraphs[pid] = paragraph

        question = self.Question(self.args, qid, qtext, para_ids)

        self.questions[qid] = question
    

def get_topk_tfidf(corpus):
    top1 = 0
    top3 = 0
    top5 = 0
    for qid in corpus.questions:


        para_scores = [(corpus.paragraphs[pid].tfidf_score, corpus.paragraphs[pid].ans_occurance) for pid in
                       corpus.questions[qid].pids]
        sorted_para_scores = sorted(para_scores, key=lambda x: x[0])
        # import pdb
        # pdb.set_trace()
        if sorted_para_scores[0][1] > 0:
            top1 += 1
        if sum([ans[1] for ans in sorted_para_scores[:3]]) > 0:
            top3 += 1
        if sum([ans[1] for ans in sorted_para_scores[:5]]) > 0:
            top5 += 1

    print(
        'top1 = {}, top3 = {}, top5 = {} '.format(top1 / len(corpus.questions), top3 / len(corpus.questions),
                                                  top5 / len(corpus.questions)))
