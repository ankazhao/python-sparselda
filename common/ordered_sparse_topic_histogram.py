#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import copy

from lda_pb2 import SparseTopicHistogramPB

class NonZero(object):

    def __init__(self, topic, count = 0):
        self.topic = topic
        self.count = count

    def __str__(self):
        return '<toipc: ' + str(self.topic) + ', count:' + str(self.count) + '>'

class OrderedSparseTopicHistogram(object):
    """OrderedSparseTopicHistogram implements the class of sparse topic
    histogram, which maintains the topics in descending orderd by their counts.
    """
    def __init__(self, num_topics):
        self.non_zeros = []  # item fmt: NonZero<topic, count>
        self.num_topics = num_topics

    def size(self):
        """Returns the size of the sparse sequence 'self.non_zeros'.
        """
        return len(self.non_zeros)

    def serialize_to_string(self):
        """Serialize the OrderedSparseTopicHistogram to SparseTopicHistogramPB
        string.
        """
        sparse_topic_hist = SparseTopicHistogramPB()
        for non_zero in self.non_zeros:
            non_zero_pb = sparse_topic_hist.non_zeros.add()
            non_zero_pb.topic = non_zero.topic
            non_zero_pb.count = non_zero.count
        return sparse_topic_hist.SerializeToString()

    def parse_from_string(self, sparse_topic_hist_str):
        """Parse OrderedSparseTopicHistogram from SparseTopicHistogramPB
        serialized string.
        """
        self.non_zeros = []
        sparse_topic_hist = SparseTopicHistogramPB()
        sparse_topic_hist.ParseFromString(sparse_topic_hist_str)
        for non_zero_pb in sparse_topic_hist.non_zeros:
            self.non_zeros.append(NonZero(non_zero_pb.topic, non_zero_pb.count))

    def count(self, topic):
        """returns the count of topic
        """
        for non_zero in self.non_zeros:
            if non_zero.topic == topic:
                return non_zero.count
        return 0

    def increase_topic(self, topic, count = 1):
        """add count on topic.
        """
        assert (topic >= 0 and topic < self.num_topics and count > 0)

        index = -1
        for i, non_zero in enumerate(self.non_zeros):
            if non_zero.topic == topic:
                non_zero.count += count
                index = i
                break;

        if index == -1:
            self.non_zeros.append(NonZero(topic, count))
            index = len(self.non_zeros) - 1

        # ensure that topics sorted by their counts.
        non_zero = copy.copy(self.non_zeros[index])
        while index > 0 and non_zero.count > self.non_zeros[index - 1].count:
            self.non_zeros[index] = copy.copy(self.non_zeros[index - 1])
            index -= 1
        self.non_zeros[index] = copy.copy(non_zero)

    def decrease_topic(self, topic, count = 1):
        """subtract count from topic.
        """
        assert (topic >= 0 and topic < self.num_topics and count > 0)

        index = -1
        for i, non_zero in enumerate(self.non_zeros):
            if non_zero.topic == topic:
                non_zero.count -= count
                assert non_zero.count >= 0
                index = i
                break;

        assert index != -1

        # ensure that topics sorted by their counts.
        non_zero = copy.copy(self.non_zeros[index])
        while index < len(self.non_zeros) - 1 and \
                non_zero.count < self.non_zeros[index + 1].count:
                    self.non_zeros[index] = \
                            copy.copy(self.non_zeros[index + 1])
                    index += 1
        if non_zero.count == 0:
            del self.non_zeros[index:]
        else:
            self.non_zeros[index] = copy.copy(non_zero)

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        topic_hist_str = []
        for non_zero in self.non_zeros:
            topic_hist_str.append(str(non_zero))
        return '\n'.join(topic_hist_str)
