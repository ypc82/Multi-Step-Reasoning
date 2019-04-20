import config
import os
import pickle
import logging
import sys
import json
from multi_corpus import MultiCorpus

logger = logging.getLogger()


def load_pickle(args, file_name):
    file_name = file_name + '.pkl'
    fpath = os.path.join(args.data_dir, args.src, "data", args.domain, file_name)
    logger.info(f"Loading pickle file from {fpath}")
    with open(fpath, "rb") as fin:
        data = pickle.load(fin)
    # import pdb; pdb.set_trace()
    return data

def get_pos_neg(paras):
    pos = 0
    neg = 0
    for key, value in paras.items():
        entails = value.ans_occurance > 0
        pos += entails
        neg += (1 - entails)
    return pos + neg, pos, neg

def main(args):
    train_exs = load_pickle(args, args.train_file_name)
    train_tot, train_pos, train_neg = get_pos_neg(train_exs.paragraphs)
    logger.info(f"Train :: Postive : {train_pos}, Negative : {train_neg}, Total : {train_tot}")
    logger.info(f"Train :: Pos % : {train_pos / train_tot}, Neg % : {train_neg / train_tot}")
    
    val_exs = load_pickle(args, args.dev_file_name)
    val_tot, val_pos, val_neg = get_pos_neg(val_exs.paragraphs)
    logger.info(f"Validation :: Positive : {val_pos}, Negative : {val_neg}, Total : {val_tot}")
    logger.info(f"Validation :: Pos $ : {val_pos / val_tot}, Neg % : {val_neg / val_tot}")
    
    test_exs = load_pickle(args, args.test_file_name)
    test_tot, test_pos, test_neg = get_pos_neg(test_exs.paragraphs)
    logger.info(f"Test :: Postive : {test_pos}, Negative : {test_neg}, Total : {test_tot}")
    logger.info(f"Test :: Pos % : {test_pos / test_tot}, Neg % {test_neg / test_tot}")

if __name__ == '__main__':
    
    logger.info('_' * 100)
    args = config.get_args()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), indent=4, sort_keys=True))

    main(args)
