#!/usr/bin/env bash
set -x

#MEM=50GB
#PARTITION=m40-long
#NGPUS=1

#python paragraph_encoder/data_reader.py 
#    --qa_file $ARC_DIR/ARC-Challenge/ARC-Challenge-Train_dgem_onlyAns_filtered.json \
#    --corpus_file $ARC_DIR/ARC_Corpus.txt \
#    --outfile_path $DATADIR/arc/data/web-open/processed_train.pkl

# Train
#srun --partition $PARTITION --gres=gpu:$NGPUS --mem=$MEM \
#python paragraph_encoder/train_para_encoder.py --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --src scitail --model_dir saved_model --experiment_name question_answer --train_file_name processed_train_no_neg --dev_file_name processed_dev_no_neg --num_epochs 100 --augment_train --min_hit_len 5 --max_hit_len 50 --neg_sample_rate 5
#python paragraph_encoder/train_para_encoder.py --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --src scitail --model_dir saved_model --experiment_name question_answer --augment_train --min_hit_len 5 --max_hit_len 50 --neg_sample_rate 3 --num_epochs 100

# Evaluation
#python paragraph_encoder/train_para_encoder.py --test_only --pretrained saved_model/20190426-190aa163/model.mdl --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --model_dir saved_model --src arc --experiment_name question_answer --num_topk_paras 10 --test_file_name processed_test --batch_size 512 --test_batch_size 512
python paragraph_encoder/train_para_encoder.py --test_only --pretrained saved_model/20190426-190aa163/model.mdl --data_dir /mnt/nfs/work1/mccallum/yipeichen/data/multi-step-reasoning/data --model_dir saved_model --src scitail --experiment_name question_answer --test_file_name processed_test
