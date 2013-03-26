#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

from lda_pb2 import NonZero
from lda_pb2 import SparseTopicHistogram

class OrderedSparseTopicHistogram(object):
    """OrderedSparseTopicHistogram implements the wrapper of
    SparseTopicHistogram, which maintains the topics sorted
    by their counts.
    """
    def __init__(self, num_topics):
        self.sparse_topic_hist = SparseTopicHistogram()
        self.num_topics = num_topics

    def size(self):
        """returns the size of the sparse sequence
        'self.sparse_topic_hist'.
        """
        return len(self.sparse_topic_hist.non_zeros)

    def count(self, topic):
        """returns the count of topic
        """
        for non_zero in self.sparse_topic_hist.non_zeros:
            if non_zero.topic == topic:
                return non_zero.count
        return 0

    def increase_topic(self, topic, count = 1):
        """add count on topic.
        """
        assert (topic >= 0 and topic < self.num_topics and count > 0)

        index = -1
        for i, non_zero in enumerate(self.sparse_topic_hist.non_zeros):
            if non_zero.topic == topic:
                non_zero.count += count
                index = i
                break;

        if index == -1:
            non_zero = self.sparse_topic_hist.non_zeros.add()
            non_zero.topic = topic
            non_zero.count = count
            index = len(self.sparse_topic_hist.non_zeros) - 1

        # ensure that topics sorted by their counts.
        non_zero = NonZero()
        non_zero.CopyFrom(self.sparse_topic_hist.non_zeros[index])
        while index > 0 and \
                non_zero.count > \
                self.sparse_topic_hist.non_zeros[index - 1].count:
                    self.sparse_topic_hist.non_zeros[index].CopyFrom( \
                            self.sparse_topic_hist.non_zeros[index - 1])
                    index -= 1
        self.sparse_topic_hist.non_zeros[index].CopyFrom(non_zero)

    def decrease_topic(self, topic, count = 1):
        """subtract count from topic.
        """
        assert (topic >= 0 and topic < self.num_topics and count > 0)

        index = -1
        for i, non_zero in enumerate(self.sparse_topic_hist.non_zeros):
            if non_zero.topic == topic:
                non_zero.count -= count
                assert non_zero.count >= 0
                index = i
                break;

        assert index != -1

        # ensure that topics sorted by their counts.
        non_zero = NonZero()
        non_zero.CopyFrom(self.sparse_topic_hist.non_zeros[index])
        while index < len(self.sparse_topic_hist.non_zeros) - 1 and \
                non_zero.count < \
                self.sparse_topic_hist.non_zeros[index + 1].count:
                    self.sparse_topic_hist.non_zeros[index].CopyFrom( \
                            self.sparse_topic_hist.non_zeros[index + 1])
                    index += 1
        if non_zero.count == 0:
            del self.sparse_topic_hist.non_zeros[index:]
        else:
            self.sparse_topic_hist.non_zeros[index].CopyFrom(non_zero)

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        return str(self.sparse_topic_hist)
