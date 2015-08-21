# -*- coding: utf-8 -*-

"""
English Word Segmentation in Python

Word segmentation is the process of dividing a phrase without spaces back
into its constituent parts. For example, consider a phrase like "thisisatest".
For humans, it's relatively easy to parse. This module makes it easy for
machines too. Use `segment` to parse a phrase into its parts:

>>> from wordsegment import segment
>>> segment('thisisatest')
['this', 'is', 'a', 'test']

In the code, 1024908267229 is the total number of words in the corpus. A
subset of this corpus is found in unigrams.txt and bigrams.txt which
should accompany this file. A copy of these files may be found at
http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt
respectively.

# Copyright (c) 2014 by Grant Jenks

Based on code from the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/

Original Copyright (c) 2008-2009 by Peter Norvig
"""

import sys
from os.path import join, dirname, realpath
from math import log10
from functools import wraps

if sys.hexversion < 0x03000000:
    range = xrange

FOUND = True
NOT_FOUND = False


class Data(object):
    def __init__(self):
        self._unigram_counts = dict()
        self._unigram_counts = self.parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))

    def parse_file(self,filename):
        with open(filename) as fptr:
            lines = (line.split('\t') for line in fptr)
            return dict((word, float(number)) for word, number in lines)

    @property
    def data(self):
        return self._unigram_counts

    def __iter__(self):
        for each in self._unigram_counts.keys():
            yield each

class Construct_corpus_tree(object):
    def __init__(self, min_length):
        self._minlen = min_length
        
    @property
    def ngram_distribution(self):
        ngram_distribution = dict()
        d = Data()
        data = d.data
        for x in d:
            if len(x) >= self._minlen:
                cut_x = x[:self._minlen]
                if cut_x in ngram_distribution:
                    ngram_distribution[cut_x] += data[x]
                    ngram_distribution
                else:
                    ngram_distribution[cut_x] = data[x]

        return ngram_distribution

    @property
    def ngram_tree(self):
        ngram_tree = dict()
        d = Data()
        for x in d:
            if len(x) >= self._minlen:
                cut_x = x[:self._minlen]
                if cut_x in ngram_tree:
                    ngram_tree[cut_x].append(x)
                else:
                    ngram_tree[cut_x] = [x]

        return ngram_tree

class heuristic_word_find(object):
    __slot__ = ("_minlen", "_casesensitive")

    def __init__(self, min_length, casesensitive = False):
        self._minlen = min_length
        self._casesensitive = casesensitive
        corpus = Construct_corpus_tree(self._minlen)
        self.ngram_distribution = corpus.ngram_distribution
        self.ngram_tree = corpus.ngram_tree

    def _divide(self):
        """
        Iterator finds ngrams and their suffix. 
        An example input of string "helloworld" yields the following tuples,
        ('hello', 'world')
        ('ellow', 'orld')
        ('llowo', 'rld')
        ('lowor', 'ld')
        ('oworl', 'd')
        ('world', '')
        """
        counter = 0
        for cut_point in range(self._minlen, len(self._string)+1):
            print (self._string[counter:cut_point], self._string[cut_point:])
            yield (self._string[counter:cut_point], self._string[cut_point:])
            counter += 1

    #public interface
    def huristic_search(self, text):
        if self._casesensitive == False:
            self._string = text.lower()
        else:
            pass #for current version, only support lowercase version

        temp_dic = dict()
        candidate_list = []
        pair_dic = dict()
        for prefix, suffix in self._divide():
            pair_dic[prefix] = suffix
            if prefix in self.ngram_distribution:
                temp_dic[prefix] = self.ngram_distribution[prefix]
                candidate_list.append((self.ngram_distribution[prefix], prefix))
            else:
                pass #means this prefix was not likely to be a part of meaningful word
        
        candidate_list.sort(reverse=True)
        for each in candidate_list:
            print "[debug]: target string --> {}".format(each[1] + pair_dic[each[1]])
            print "[debug]: ngram tree    --> {}".format(self.ngram_tree[each[1]])
            if each[1] in self.ngram_tree[each[1]]:
                print "Found: {}".format(each[1])
                return FOUND
            else:
                for x in self.ngram_tree[each[1]]:
                    if x in each[1] + pair_dic[each[1]]:
                        print "Found: {}".format(x)
                        return FOUND

        return NOT_FOUND

    #public interface
    def get_meaningful_words(self, text):
        if self._casesensitive == False:
            self._string = text.lower()
        else:
            pass #for current version, only support lowercase version
        temp_dic = dict()
        candidate_list = []
        pair_dic = dict()
        for prefix, suffix in self._divide():
            pair_dic[prefix] = suffix
            if prefix in self.ngram_distribution:
                temp_dic[prefix] = self.ngram_distribution[prefix]
                candidate_list.append((self.ngram_distribution[prefix], prefix))
            else:
                pass #means this prefix was not likely to be a part of meaningful word
        
        candidate_list.sort(reverse=True)
        meaningful_words = []
        for each in candidate_list:
            print "[debug]: target string --> {}".format(each[1] + pair_dic[each[1]])
            print "[debug]: ngram tree    --> {}".format(self.ngram_tree[each[1]])
            if each[1] in self.ngram_tree[each[1]]:
                meaningful_words.append(each[1])
#            else:
#                for x in self.ngram_tree[each[1]]:
#                    if x in each[1] + pair_dic[each[1]]:
#                        meaningful_words.append(x)

        return meaningful_words


