#!/usr/bin/env  python
# coding=utf-8

# author: Lifeng Wang (ofandywang@gmail.com)
# date: 2013.03.18
# project: python-sparselda

import unittest

from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram

class OrderedSparseTopicHistogramTest(unittest.TestCase):

    def setUp(self):
        self.num_topics = 20
        self.ordered_sparse_topic_hist = \
                OrderedSparseTopicHistogram(self.num_topics)
        for i in xrange(0, 10):
            self.ordered_sparse_topic_hist.increase_topic(i, i + 1)

    def test_ordered_sparse_topic_hist(self):
        topic_hist = self.ordered_sparse_topic_hist.sparse_topic_hist
        self.assertEqual(10, len(topic_hist.non_zeros))
        for i in xrange(0, len(topic_hist.non_zeros)):
            self.assertEqual(10 - i - 1, topic_hist.non_zeros[i].topic)
            self.assertEqual(10 - i, topic_hist.non_zeros[i].count)

    def test_num_topics(self):
        self.assertEqual(self.num_topics, \
                self.ordered_sparse_topic_hist.num_topics)

    def test_size(self):
        self.assertEqual(10, self.ordered_sparse_topic_hist.size())

    def test_count(self):
        for i in xrange(0, 10):
            self.assertEqual(i + 1, \
                    self.ordered_sparse_topic_hist.count(i))
        for i in xrange(10, 20):
            self.assertEqual(0, \
                    self.ordered_sparse_topic_hist.count(i))

    def test_increase_topic(self):
        for i in xrange(0, 20):
            self.ordered_sparse_topic_hist.increase_topic(i, i + 1)

            topic_hist = self.ordered_sparse_topic_hist.sparse_topic_hist
            for j in xrange(0, len(topic_hist.non_zeros) - 1):
                self.assertGreaterEqual( \
                        topic_hist.non_zeros[j].count, \
                        topic_hist.non_zeros[j + 1].count)

        self.assertEqual(2, self.ordered_sparse_topic_hist.count(0))
        self.assertEqual(12, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(11, self.ordered_sparse_topic_hist.count(10))
        self.assertEqual(16, self.ordered_sparse_topic_hist.count(15))

    def test_decrease_topic(self):
        self.assertEqual(6, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(7, self.ordered_sparse_topic_hist.count(6))
        self.ordered_sparse_topic_hist.decrease_topic(5, 1)
        self.ordered_sparse_topic_hist.decrease_topic(6, 4)
        self.assertEqual(10, self.ordered_sparse_topic_hist.size())
        self.assertEqual(5, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(3, self.ordered_sparse_topic_hist.count(6))

        topic_hist = self.ordered_sparse_topic_hist.sparse_topic_hist
        for i in xrange(0, len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual( \
                    topic_hist.non_zeros[i].count, \
                    topic_hist.non_zeros[i + 1].count)

        self.ordered_sparse_topic_hist.decrease_topic(6, 3)
        self.assertEqual(9, self.ordered_sparse_topic_hist.size())
        for i in xrange(0, len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual( \
                    topic_hist.non_zeros[i].count, \
                    topic_hist.non_zeros[i + 1].count)

if __name__ == '__main__':
    unittest.main()
