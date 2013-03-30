#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import argparse
import logging
import os
import random

from common.model import Model
from common.vocabulary import Vocabulary
from training.sparselda_train_gibbs_sampler import SparseLDATrainGibbsSampler
from training.model_evaluator import ModelEvaluator

def main(args):
    model = Model(args.num_topics, args.topic_prior, args.word_prior)
    vocabulary = Vocabulary()
    vocabulary.load(args.vocabulary_file)
    sparselda_train_gibbs_sampler = \
            SparseLDATrainGibbsSampler(model, vocabulary)
    sparselda_train_gibbs_sampler.load_corpus(args.corpus_dir)

    rand = random.Random()

    for i in xrange(args.total_iterations):
        logging.info('sparselda trainer, gibbs sampling iteration %d.' % (i + 1))
        sparselda_train_gibbs_sampler.gibbs_sampling(rand)

        # dump lda model
        if (i + 1) % args.save_model_interval == 0:
            logging.info('iteration %d start saving lda model.' % (i + 1))
            sparselda_train_gibbs_sampler.save_model( \
                    args.model_dir, i + 1)
            topic_words_stat = TopicWordsStat(mode, vocabulary)
            topic_words_stat.save( \
                    args.model_dir + '/topic_top_words.%d' % (i + 1),
                    args.topic_word_accumalated_prob_threshold)
            logging.info('iteration %d save lda model ok.' % (i + 1))

        # dump checkpoint
        if args.is_save_checkpoint and \
                (i + 1) % args.save_checkpoint_interval == 0:
            logging.info('iteration %d start saving checkpoint.' % (i + 1))
            sparselda_train_gibbs_sampler.save_checkpoint( \
                    args.checkpoint_dir, i + 1)
            logging.info('iteration %d save checkpoint ok.' % (i + 1))

        # compute the loglikelihood
        if (i + 1) % args.compute_loglikelihood_interval == 0:
            logging.info('iteration %d start computing loglikelihood.' % (i + 1))
            model_evaluator = ModelEvaluator(model, vocabulary)
            ll = model_evaluator.compute_loglikelihood( \
                    sparselda_train_gibbs_sampler.documents)
            logging.info('iteration %d loglikelihood is %f.' % (i + 1, ll))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SparseLDA trainer.')
    parser.add_argument('--corpus_dir', required=True, \
            help='the corpus directory.')
    parser.add_argument('--vocabulary_file', required=True, \
            help='the vocabulary file.')
    parser.add_argument('--num_topics', type=int, required=True, \
            help='the num of topics.')
    parser.add_argument('--topic_prior', type=float, default=0.1, \
            help='the topic prior alpha.')
    parser.add_argument('--word_prior', type=float, default=0.01, \
            help='the word prior beta.')
    parser.add_argument('--total_iterations', type=int, default=10000, \
            help='the total iteration.')
    parser.add_argument('--model_dir', required=True, \
            help='the model directory.')
    parser.add_argument('--save_model_interval', type=int, default=100, \
            help='the interval of save_model action.')
    parser.add_argument('--topic_word_accumalated_prob_threshold', type=float, \
            default=0.8, help='the accumalated_prob_threshold of topic words.')
    parser.add_argument('--is_save_checkpoint', action='store_true', \
            help='whether or not to save checkpoint.')
    parser.add_argument('--save_checkpoint_interval', type=int, default=10, \
            help='the interval of save_checkpoint action.')
    parser.add_argument('--checkpoint_dir', help='the checkpoint directory.')
    parser.add_argument('--compute_loglikelihood_interval', type=int, \
            default=10, help='the interval of compute_loglikelihood action.')

    args = parser.parse_args()
    logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'), \
            level = logging.DEBUG, filemode = 'w', \
            format = '%(asctime)s - %(levelname)s: %(message)s')

    main(args)
