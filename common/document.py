#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import random
from lda_pb2 import DocumentPB
from model import Model
from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram
from vocabulary import Vocabulary

class Document(object):
    """
    """
    def __init__(self, num_topics):
        self.document_pb = None  # word occurances of the document,
                                 # item fmt: Word {id, topic}
        self.doc_topic_hist = None  # N(z|d), OrderedSparseTopicHistogram
        self.num_topics = num_topics

    def parse_from_tokens(self, doc_tokens, rand, vocabulary, model = None):
        """Parse the text document from tokens. Only tokens in vocabulary
        and model will be considered.
        """
        self.document_pb = DocumentPB()
        self.doc_topic_hist = OrderedSparseTopicHistogram(self.num_topics)

        for token in doc_tokens:
            word_index = vocabulary.word_index(token)
            if word_index != -1 and \
                    (model == None or model.has_word(word_index)):
                # initialize a random topic for cur word
                word = self.document_pb.words.add()
                word.id = word_index
                word.topic = rand.randint(0, self.num_topics - 1)
                self.doc_topic_hist.increase_topic(word.topic, 1)

    def serialize_to(self):
        """Serialize document to DocumentPB string.
        """
        return self.document_pb.SerializeToString()

    def parse_from_string(self, document_str):
        """Parse document from DocumentPB serialized string.
        """
        self.document_pb = DocumentPB()
        self.document_pb.ParseFromString(document_str)
        self.doc_topic_hist = OrderedSparseTopicHistogram(self.num_topics)
        for word in self.document_pb.words:
            self.increase_topic(word.topic, 1)

    def num_words(self):
        return len(self.document_pb.words)

    def get_topic_count(self, topic):
        """Returns N(z|d).
        """
        return self.doc_topic_hist.count(topic)

    def increase_topic(self, topic, count = 1):
        self.doc_topic_hist.increase_topic(topic, count)

    def decrease_topic(self, topic, count = 1):
        self.doc_topic_hist.decrease_topic(topic, count)

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        return str(self.document_pb)

