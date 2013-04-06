#!/bin/bash

nohup python lda_trainer.py \
    --corpus_dir=testdata/corpus \
    --vocabulary_file=testdata/vocabulary.dat \
    --num_topics=500 \
    --topic_prior=0.1 \
    --word_prior=0.01 \
    --save_model_interval=10 \
    --model_dir=sparselda_models \
    --save_checkpoint_interval=10 \
    --checkpoint_dir=sparselda_checkpoints \
    --total_iterations=10000 \
    --compute_loglikelihood_interval=10 \
    --topic_word_accumulated_prob_threshold=0.5 \
    > train.log 2>&1 &

