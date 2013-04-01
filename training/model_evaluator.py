#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import logging
import math
import os
import random
import sys

sys.path.append('..')
from common.document import Document
from common.model import Model
from common.vocabulary import Vocabulary

class ModelEvaluator(object):
    """ModelEvaluator implements the evaluation method of lda model's quality.
    """

    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary

        # cache matrix p(w|z), indexed by word.
        self.word_topic_dist = \
                self.model.get_word_topic_dist(self.vocabulary.size())

    def compute_loglikelihood(self, documents):
        """Compute and return the loglikelihood of documents.

        p(D|M) = p(d1)p(d2)...

        p(d) = p(w1)p(w2)...
             = sum_z {p(z|d)p(w1|z)} * sum_z {p(z|d)p(w2|z)} * ...

        log(p(d)) = log(sum_z p(z|d)p(w1|z)) + log(sum_z p(z|d)p(w2|z)) + ...

        p(D|M) -> log(p(D|M)) = log(p(d1)) + log(p(d2)) + ...
        """
        loglikelihood = 0.0
        for document in documents:
            doc_dense_topic_dist = self._compute_doc_topic_distribution(document)
            doc_loglikelihood = 0.0
            for word in document.words:
                if word.id not in self.model.word_topic_hist:
                    continue
                word_topic_dist = self.word_topic_dist[word.id]
                for topic, prob in enumerate(word_topic_dist):
                    doc_loglikelihood += \
                            math.log(prob * doc_dense_topic_dist[topic])
            loglikelihood += doc_loglikelihood
        return loglikelihood

    def _compute_doc_topic_distribution(self, document):
        topic_hist_sum = 0
        for non_zero in document.doc_topic_hist.non_zeros:
            topic_hist_sum += non_zero.count

        dense_topic_dist = []
        for i in xrange(self.model.num_topics):
            dense_topic_dist.append( \
                    self.model.hyper_params.topic_prior * \
                    document.get_topic_count(i) / \
                    self.model.hyper_params.topic_prior * self.model.num_topics + \
                    topic_hist_sum)
        return dense_topic_dist
