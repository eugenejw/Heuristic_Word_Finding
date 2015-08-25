# -*- coding: utf-8 -*-

"""
English Word Heuristical Finder in Python

Heuristical Word Finding is the process of find meaningful word(s), which
contains at least N characters(len(word)>=N), from a string.

It provides two interfaces,
1. find whether a string contains meaningful word,
   if found, terminate the search, and return True.

>>> from heuristic_word_finding import WordFind
>>> query = WordFind(5)
>>> query.search("afwsfjacertificatequwr        legalehgfq#$rehgqu^^niver352")
True

2. find all meaningful words, whose len(word)>= N, from the string.
   If found, return word(s) found in a list.
>>> from heuristic_word_finding import WordFind
>>> query = WordFind(5)
>>> query.get("afwsfjacertificatequwr   legalehgfq#$rehgqu^^niver352")
['legal', 'certificate']

In the code, a modified version of  corpus containing 1024908267229 total
number of words is used. 
I've truncated the corpus by removing all non-dictionary words.
The dictionary I used to truncate it is Python Enchant Dictionary.
So any word found in the truncated corpus will be deemed as meaningful.
The probability of words in the corpus is used to achieve the "heuristic"
The search will start from the most popular words to make it
terminate earlier.

The original version of the corpus  may be found at
http://norvig.com/ngrams/ under the name count_1w.txt.

# Copyright (c) 2015 by Weihan Jiang

Original corpus is from the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/

Original Copyright (c) 2008-2009 by Peter Norvig
"""

import sys
from os.path import join, dirname, realpath

FOUND = True
NOT_FOUND = False

if sys.hexversion < 0x03000000:
    range = xrange

def parse_file(filename):
    '''
    Global function that parses file and form a dictionary.
    '''
    with open(filename) as fptr:
        lines = (line.split('\t') for line in fptr)
        return dict((word, float(number)) for word, number in lines)

class Data(object):
    '''
    Read corpus from path, and provide the following functionalities,
    1. data as "property", it is a dictionary where key is word,
       while the value is the frequency count of this word.
    2. generator that yield word and its frequency
    '''
    def __init__(self):
        self._unigram_counts = dict()
        self._unigram_counts = parse_file(
		join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt')
	)

    @property
    def data(self):
        '''
        return the whole dictionary out to user as a property.
        '''
        return self._unigram_counts

    def __iter__(self):
        for each in self._unigram_counts.keys():
            yield each

class ConstructCorpus(object):
    '''
    according to the minimal character limit,
    construct a corpus at initial time.
    it provides the following two properties,
    1. ngram_distribution -- a dictionary where key is the ngram,
       value is an int of summation of frequency of each English
       word starts with that specific ngram.
    2. ngram_tree -- a dictionary where key is the ngram,
       value is a list containing all possile English word
       starts with that specific ngram.
    '''
    def __init__(self, min_length):
        self._minlen = min_length

    @property
    def ngram_distribution(self):
        '''
        return a dictionary containing the following pairs,
        key: ngram string, for example, when minlen=5,
             the ngram string for word "university" is "unive".
        value: added-up frequency(from google corpus) of all
               words starting with "unive".
        '''
        ngram_distribution = dict()
        instance_d = Data()
        data = instance_d.data
        for entry in instance_d:
            if len(entry) >= self._minlen:
                cut = entry[:self._minlen]
                if cut in ngram_distribution:
                    ngram_distribution[cut] += data[entry]
#                    ngram_distribution
                else:
                    ngram_distribution[cut] = data[entry]

        return ngram_distribution

    @property
    def ngram_tree(self):
        '''
        return a dictionary containing the following pairs,
        key: ngram string, for example, when minlen=5,
             the ngram string for word "university" is "unive".
        value: all words starting with the ngram,
               in the example, it is "unive".
        '''
        ngram_tree = dict()
        instance_d = Data()
        for entry in instance_d:
            if len(entry) >= self._minlen:
                cut = entry[:self._minlen]
                if cut in ngram_tree:
                    ngram_tree[cut].append(entry)
                else:
                    ngram_tree[cut] = [entry]

        return ngram_tree

class WordFiner(object):
    '''
    class that provides the following two fuctions,
    1. Finds whether a string contains any meaningful word,
       that is more than <minimal length> characters.
       If found, return True.
    2. Finds all meaningful words in a string.
    '''

    __slot__ = ("_minlen", "_casesensitive")

    def __init__(self, min_length, casesensitive=False):
        self._minlen = min_length
        self._string = ''
        self._casesensitive = casesensitive
        corpus = ConstructCorpus(self._minlen)
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
            yield (self._string[counter:cut_point], self._string[cut_point:])
            counter += 1

    #public interface I
    def search(self, text):
        '''
        Public interface,
        for user to query whether a string contains meaningful
        word whose len(word) >= minlen(set in initial phase of the class)
        '''
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
                #means this prefix was not likely
                #to be a part of meaningful word.
                pass

        candidate_list.sort(reverse=True)
        for each in candidate_list:
            if each[1] in self.ngram_tree[each[1]]:
                print "Found: {}".format(each[1])
                return FOUND
            else:
                for word in self.ngram_tree[each[1]]:
                    if word in each[1] + pair_dic[each[1]]:
                        print "Found: {}".format(word)
                        return FOUND

        return NOT_FOUND

    #public interface II
    def get(self, text):
        '''
        Public interface for user to find all meaningful words in a string.
        '''
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
                candidate_list.append(
                    (self.ngram_distribution[prefix], prefix)
                )
            else:
                #means this prefix was not likely
                #to be a part of meaningful word
                pass

        candidate_list.sort(reverse=True)
        meaningful_words = []
        for each in candidate_list:
            for word in self.ngram_tree[each[1]]:
                if word in each[1] + pair_dic[each[1]]:
#                    print "{} found".format(word)
                    meaningful_words.append(word)

        return meaningful_words
