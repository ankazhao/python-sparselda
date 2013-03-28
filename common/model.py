#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import logging
import os
from lda_pb2 import SparseTopicHistogram
from lda_pb2 import GlobalTopicHistogram
from lda_pb2 import WordTopicHistogram
from lda_pb2 import HyperParams
from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram
from recordio import RecordReader
from recordio import RecordWriter
from vocabulary import Vocabulary

class Model(object):
    """Model implements the sparselda model.
    It includes the following parts:
        0. num_topics, represents |K|.
        1. global_topic_hist, represents N(z).
        2. word_topic_hist, represents N(w|z).
        3. hyper_params
           3.1 topic_prior, represents the dirichlet prior of topic \alpha.
           3.2 word_prior, represents the dirichlet prior of word \beta.
    """
    GLOABLE_TOPIC_HIST_FILENAME = "lda.global_topic_hist"
    WORD_TOPIC_HIST_FILENAME = "lda.word_topic_hist"
    HYPER_PARAMS_FILENAME = "lda.hyper_params"

    def __init__(self, num_topics, topic_prior = 0.01, word_prior = 0.1):
        self.num_topics = num_topics

        self.global_topic_hist = GlobalTopicHistogram()  # item fmt: N(z)
        for i in xrange(0, self.num_topics):
            self.global_topic_hist.topic_counts.append(i)

        self.word_topic_hist = {}  # item fmt: w -> N(w|z)

        # TODO(fandywang): optimize the hyper_params.
        # Because we find that an asymmetric Dirichlet prior over the document-
        # topic distributions has substantial advantages over a symmetric prior,
        # while an asymmetric prior over topic-word distributions provides no
        # real benefit.
        # See 'Hanna Wallach, David Mimno, and Andrew McCallum. 2009.
        # Rethinking LDA: Why priors matter. In Proceedings of NIPS-09,
        # Vancouver, BC.' for more details.
        self.hyper_params = HyperParams()
        self.hyper_params.topic_prior = topic_prior  # alpha, default symmetrical
        self.hyper_params.word_prior = word_prior  # beta, default symmetrical

    def save(self, model_dir):
        logging.info('Save lda model to %s.' % model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self._save_word_topic_hist(model_dir + "/" + \
                self.__class__.WORD_TOPIC_HIST_FILENAME)
        self._save_global_topic_hist(model_dir + "/" + \
                 self.__class__.GLOABLE_TOPIC_HIST_FILENAME)
        self._save_hyper_params(model_dir + "/" + \
                 self.__class__.HYPER_PARAMS_FILENAME)

    def load(self, model_dir):
        logging.info('Load lda model from %s.' % model_dir)
        assert self._load_global_topic_hist(model_dir + "/" + \
                self.__class__.GLOABLE_TOPIC_HIST_FILENAME)
        self.num_topics = len(self.global_topic_hist.topic_counts)
        assert self._load_word_topic_hist(model_dir + "/" + \
                self.__class__.WORD_TOPIC_HIST_FILENAME)
        assert self._load_hyper_params(model_dir + "/" + \
                self.__class__.HYPER_PARAMS_FILENAME)

    def _save_global_topic_hist(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        record_writer.write(self.global_topic_hist.SerializeToString())
        fp.close()

    def _save_word_topic_hist(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        for word, ordered_sparse_topic_hist in \
                self.word_topic_hist.items():
            word_topic_hist_pb = WordTopicHistogram()
            word_topic_hist_pb.word = word
            word_topic_hist_pb.sparse_topic_hist.CopyFrom( \
                    ordered_sparse_topic_hist.sparse_topic_hist)
            record_writer.write(word_topic_hist_pb.SerializeToString())
        fp.close()

    def _save_hyper_params(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        record_writer.write(self.hyper_params.SerializeToString())
        fp.close()

    def _load_global_topic_hist(self, filename):
        logging.info('Loading global_topic_hist vector N(z).')
        self.global_topic_hist.Clear()

        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        blob = record_reader.read()
        fp.close()
        if blob == None:
            logging.error('GlobalTopicHist is nil, file %s' % filename)
            return False

        self.global_topic_hist.ParseFromString(blob)
        return True

    def _load_word_topic_hist(self, filename):
        logging.info('Loading word_topic_hist matrix N(w|z).')
        self.word_topic_hist.clear()

        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        while True:
            blob = record_reader.read()
            if blob == None:
                break
            word_topic_hist_pb = WordTopicHistogram()
            word_topic_hist_pb.ParseFromString(blob)

            ordered_sparse_topic_hist = \
                    OrderedSparseTopicHistogram(self.num_topics)
            ordered_sparse_topic_hist.sparse_topic_hist.CopyFrom( \
                    word_topic_hist_pb.sparse_topic_hist)
            self.word_topic_hist[word_topic_hist_pb.word] = \
                    ordered_sparse_topic_hist

        fp.close()
        return (len(self.word_topic_hist) > 0)

    def _load_hyper_params(self, filename):
        logging.info('Loading hyper_params topic_prior and word_prior.')
        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        blob = record_reader.read()
        fp.close()
        if blob == None:
            logging.error('HyperParams is nil, file %s' % filename)
            return False

        self.hyper_params.ParseFromString(blob)
        return True

    def has_word(self, word):
        return word in self.word_topic_hist

    def get_word_topic_dist(self, vocab_size):
        """Returns topic-word distributions matrix p(w|z), indexed by word.
        """
        word_topic_dist = {}

        # TODO(fandywang): only cache submatrix p(w|z) of some frequency words.
        for word_id, ordered_sparse_topic_hist in self.word_topic_hist.items():
            dense_topic_dist = []
            for topic in xrange(0, self.num_topics):
                dense_topic_dist.append(self.hyper_params.word_prior / \
                        (self.hyper_params.word_prior * vocab_size + \
                        self.global_topic_hist.topic_counts[topic]))

            for non_zero in ordered_sparse_topic_hist.sparse_topic_hist.non_zeros:
                dense_topic_dist[non_zero.topic] = \
                        (self.hyper_params.word_prior + non_zero.count) / \
                        (self.hyper_params.word_prior * vocab_size + \
                        self.global_topic_hist.topic_counts[topic])
            word_topic_dist[word_id] = dense_topic_dist

        return word_topic_dist

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        model_str = []
        model_str.append('num_topics: %d' % self.num_topics)
        model_str.append('GlobalTopicHist: ')
        for i in xrange(0, len(self.global_topic_hist.topic_counts)):
            model_str.append('topic: %d\tcount: %d' \
                    % (i, self.global_topic_hist.topic_counts[i]))
        model_str.append('WordTopicHist: ')
        for word, ordered_sparse_topic_hist in self.word_topic_hist.items():
            model_str.append('word: %d' % word)
            model_str.append('topic_hist: %s' % str(ordered_sparse_topic_hist))
        model_str.append('HyperParams: ')
        model_str.append(str(self.hyper_params))
        return '\n'.join(model_str)
