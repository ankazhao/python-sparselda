#!/bin/bash

nohup python lda_trainer.py \
    --corpus_dir=../../wenwen_title_800w/corpus_test \
    --vocabulary_file=../../wenwen_title_800w/vocab.txt \
    --num_topics=100 \
    --topic_prior=0.1 \
    --word_prior=0.01 \
    --save_model_interval=10 \
    --model_dir=sparselda_models \
    --save_checkpoint_interval=10 \
    --checkpoint_dir=sparselda_checkpoints \
    --total_iterations=10000 \
    --compute_loglikelihood_interval=10 \
    --topic_word_accumulated_prob_threshod=0.5 \
    > train.log 2>&1 &

