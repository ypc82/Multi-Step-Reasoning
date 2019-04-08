#!/usr/bin/env bash
set -x

MEM=75GB
PARTITION=m40-long
NGPUS=1

#python paragraph_encoder/data_reader.py 
#    --qa_file $ARC_DIR/ARC-Challenge/ARC-Challenge-Train_dgem_onlyAns_filtered.json \
#    --corpus_file $ARC_DIR/ARC_Corpus.txt \
#    --outfile_path $DATADIR/arc/data/web-open/processed_train.pkl

# Train
#python paragraph_encoder/train_para_encoder.py --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --src scitail --model_dir saved_model --experiment_name question_answer

# Evaluation
srun --partition $PARTITION --gres=gpu:$NGPUS --mem=$MEM \
python paragraph_encoder/train_para_encoder.py --test_only --pretrained saved_model/20190406-568de4e6/model.mdl --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --model_dir saved_model --src arc --experiment_name question_answer --num_topk_paras 10 --test_file_name processed_test_1K --batch_size 512 --test_batch_size 512
