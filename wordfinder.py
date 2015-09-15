# -*- coding: utf-8 -*-

"""
English Word Heuristical Finder in Python

Heuristical Word Finding is the process of find meaningful word(s), which
contains at least N characters(len(word)>=N), from a string.

It provides two interfaces,
1. find whether a string contains meaningful word,
   if found, terminate the search, and return True.

>>> from wordfinder import WordFind
>>> wf = WordFind(5)
>>> wf.search("afwsfjacertificatequwr        legalehgfq#$rehgqu^^niver352")
True

2. find all meaningful words, whose len(word)>= N, from the string.
   If found, return word(s) found in a list.
>>> from wordfinder import WordFind
>>> wf = WordFind(5)
>>> wf.get("afwsfjacertificatequwr   legalehgfq#$rehgqu^^niver352")
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
from operator import itemgetter
from itertools import groupby
import networkx as nx
from wordsegment import segment
from itertools import groupby, count
from math import log10

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

def as_range(g):
    l = list(g)
    return l[0], l[-1]

unigram_counts = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))


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

class WordFinder(object):
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
#                print ("Found: {}").format(each[1])
                return FOUND
            else:
                for word in self.ngram_tree[each[1]]:
                    if word in each[1] + pair_dic[each[1]]:
#                        print ("Found: {}").format(word)
                        return FOUND

        return NOT_FOUND

    #public interface II
    def get(self, text):
        '''
        Public interface for user to find all meaningful words in a string.
        Input: string
        Return: a set of meaningful English words found
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
                    meaningful_words.append(word)

        return set(meaningful_words)

class WordSegment(object):
    '''
    class that provides the following two fuctions,
    1. Finds whether a string contains any meaningful word,
       that is more than <minimal length> characters.
       If found, return True.
    2. Finds all meaningful words in a string.
    '''

    def __init__(self, min_length=2, casesensitive=False):
        self._minlen = min_length
        self._string = ''
        self._casesensitive = casesensitive
        corpus = ConstructCorpus(self._minlen)
        self.ngram_distribution = corpus.ngram_distribution
        self.ngram_tree = corpus.ngram_tree

    def _divide(self):
        """
        Iterator finds ngrams(with its position in string) and their suffix.
        An example input of string "helloworld" yields the following tuples,
        (('hello',(0,5)), 'world')
        (('ellow',(1,6)), 'orld')
        (('llowo',(2,7)), 'rld')
        (('lowor',(3,8)), 'ld')
        (('oworl',(4,9)), 'd')
        (('world',(5,10)), '')
        """
        counter = 0
        for cut_point in range(self._minlen, len(self._string)+1):
            yield ((self._string[counter:cut_point],(counter, counter+self._minlen)), self._string[cut_point:])
            counter += 1

    def _intersect(self, tuple_0, tuple_1):
        '''
        finds intersection of two words
        '''
        x = range(tuple_0[0],tuple_0[1])
        y = range(tuple_1[0],tuple_1[1])
        xs = set(x)
        return xs.intersection(y)
        
    '''
    def _group(self, meaningful_words):
        #[((0,5), "hello"),((0,10), "helloword"),...]
        for each in meaningful_words:
            
        pos_list = 
        for k, g in groupby(enumerate(data), lambda (i,x):i-x):
            print map(itemgetter(1), g)
    '''

    def _connected_components(self, neighbors):
        seen = set()
        def component(node):
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= neighbors[node] - seen
            yield node
        for node in neighbors:
            if node not in seen:
                yield component(node)

    def _init_graph(self, meaningful_words):
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        G = nx.Graph()
        G.add_nodes_from(meaningful_words)
        for each in meaningful_words:
            for each_2 in meaningful_words:
                if each == each_2:
                    continue
                elif self._intersect(each[0], each_2[0]):
                    if (each[0], each_2[0]) in G.edges():
                        continue
                    else:
                        G.add_edge(each, each_2)
        return G
    
    def _score(self, node_list):
        #[((0, 3),"face"),((4, 6),"boo")]
        #input, list of nodes
        #return format (score, list of nodes)
        pass
        '''
        start_pos = node_list[0][0]  #0
        end_pos = node_list[-1][0]  #6
        original_string = self._string[start_pos:end_pos]
        
        #prefer longer words
        node_len = 0
        for each in node_list:
            node_len += len(each[1])
        ave_word_len = float(node_len/len(node_list))
        '''

    def _score_by_len(self, lst):
        """Score a `word` in the context of the previous word, `prev`."""


        words = []
        score = 0
        if isinstance(lst, tuple):
            words = [lst[1]]
        else:
            for each in lst:
                words.append(each[1])

        print "words are {}\n".format(words)
        for word in words:
            if word in unigram_counts:
                score = score + len(word)
            else:
                score = score + len(word)

        return score


    def score_by_prob(lst):
        """Score a `word` in the context of the previous word, `prev`."""


        words = []
        score = 0
        if isinstance(lst, tuple):
            words = [lst[1]]
        else:
            for each in lst:
                words.append(each[1])

        print "words are {}\n".format(words)
        for word in words:
            if word in unigram_counts:
                score = score + log10((unigram_counts[word] / 1024908267229.0))
            else:
                score = score +  log10((10.0 / (1024908267229.0 * 10 ** len(word))))

        return score

    def score1(self, lst):
        """Score a `word` in the context of the previous word, `prev`."""


        words = []
        score = 0
        for each in lst:
                words.append(each[1])

        print "words are {}\n".format(words)
        for word in words:
            if word in unigram_counts:
                score = score + log10((unigram_counts[word] / 1024908267229.0))
            else:
                score = score +  log10((10.0 / (1024908267229.0 * 10 ** len(word))))

        return score



    def _max(self, lst):
        print "debug: input list is {}".format(lst)
        tmp_lst = []
        for each in lst:
            tmp_lst.append(each[0])

        max_score = max(tmp_lst)

        winners = []
        for each in lst:
            if each[0] == max_score:

                winners.append((self.score1(each[1]), each[0], each[1]))

        print winners
        print "_max:winner is {}".format(max(winners))
        return (max(winners)[1], max(winners)[2])

    def _max_2(self, lst, start_pos, end_pos):
        print "debug: input list is {}".format(lst)
        tmp_lst = []
        for each in lst:
            tmp_lst.append(each[0])

        max_score = max(tmp_lst)

        winners = []
        for each in lst:
        #    if each[0] == max_score:
               
                winners.append((self.score1(each[1]) + self._penalize(each[1], start_pos, end_pos), each[0], each[1]))

        print winners
        print "_max_2:winner is {}".format(max(winners))
        return (max(winners)[1], max(winners)[2])



    def _find_components(self, meaningful_words):
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        G = nx.Graph()
        G = self._init_graph(meaningful_words)
        components = []
        components = list(nx.connected_component_subgraphs(G))
        return components

    def _penalize(self, lst, start_pos, end_pos):
        #(0.00011509110207386408, [((1, 3), 'ac'), ((7, 11), 'king')])
        #(-8.483260969067402, [((1, 3), 'ac'), ((7, 11), 'king'), ((11, 13), 'ir')])
        if not lst[0][0][0] - start_pos == 0:
            starting_pos_penalty = log10(10.0 / (1024908267229.0 * 10 ** (lst[0][0][0] - start_pos)))
        else:
            starting_pos_penalty = 0

        if not end_pos - lst[-1][0][1] == 0:
            ending_pos_penalty = log10(10.0 / (1024908267229.0 * 10 ** (end_pos - lst[-1][0][1])))
        else:
            ending_pos_penalty = 0

        interval_penalty = [0]
        count = len(lst)
        for i in xrange(count-1):
            if not lst[i+1][0][0] - lst[i][0][1] == 0:
                interval_penalty.append(log10(10.0 / (1024908267229.0 * 10 ** (lst[i+1][0][0] - lst[i][0][1]))))
        return sum(interval_penalty) + starting_pos_penalty + ending_pos_penalty
            
            
        

    def _component_optimizing(self,component):
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        start_pos_lst = []
        end_pos_lst = []
        segment_word_lst = []
        candidate_lst = []
        for each in component:
            start_pos_lst.append(each[0][0])
            end_pos_lst.append(each[0][1])
        start_pos = min(start_pos_lst)
        end_pos = max(end_pos_lst)
        original_word = self._string[start_pos:end_pos]
        #if the component contains only one word
        if len(component) == 1:
            print "debug: component is {}\n".format(component.nodes())
            return ((start_pos, end_pos), [component.nodes()[0][1]])
        #if one of the words has len==component
        candidate_list = []
        candidate_list = self.opt(component)
        scored_candidate_list = []
        for each in candidate_list:
            scored_candidate_list.append(self._segment(each))
            print self._segment(each)
        print "##################\n"
        print "start_pos: {}".format(start_pos)
        print "end_pos: {}".format(end_pos)
        print "scored_candidate_list:"
        for each in scored_candidate_list:
            print each
        '''
        penaltized_list = []
        for each in scored_candidate_list:
            #(0.00011509110207386408, [((1, 3), 'ac'), ((7, 11), 'king')])
            penaltized_list.append([each[0] + self._penalize(each, start_pos, end_pos), each[1]])
            print "scored candidare: {0} - {1}".format(each, self._penalize(each, start_pos, end_pos))
        
        for each in penaltized_list:
            print "penalized candidare: {}".format(each)


        '''
        
#        print "MAX score is: {}\n".format(self._max_2(scored_candidate_list, start_pos, end_pos))
        #MAX score is: (9, [((4, 11), 'booking'), ((11, 13), 'ir')])
        scored_candidate_list.sort(reverse=True)
        max_3_list = scored_candidate_list[:3]
        max_result = []
        max_result = self._max_2(max_3_list, start_pos, end_pos)
        for each in max_result[1]:
            segment_word_lst.append(each[1])

#        if max(scored_candidate_list)[1][0][0] != start_pos:
        start_pos = max_result[1][0][0][0]
        end_pos = max_result[1][-1][0][1]
        return ((start_pos, end_pos), segment_word_lst)

    def _score(self, lst):
        """Score a `word` in the context of the previous word, `prev`."""


        words = []
        score = 0
        if isinstance(lst, tuple):
            words = [lst[1]]
        else:
            for each in lst:
                words.append(each[1])

        print "words are {}\n".format(words)
        for word in words:
            if word in unigram_counts:
                score = score + (unigram_counts[word] / 1024908267229.0)
            else:
                score = score +  (10.0 / (1024908267229.0 * 10 ** len(word)))

        return score

    def _segment(self, lst):
        """return a list of words that is the best segmentation of 'text'"""
        print "debug: lst is :{}\n".format(lst)
        def search(lst):

            flag = "ALL_NON_LIST"
            if len(lst) == 1 and not isinstance(lst[0], list):
                print "returned {}\n".format(lst)

                return (self._score_by_len(lst), lst)
            for each in lst:
                if isinstance(each, list):
                    flag = "LIST_FOUND"
            if flag == "ALL_NON_LIST":
                print "returned {}\n".format(lst)
                return (self._score_by_len(lst), lst)

            def candidates():
                leading_word = lst[0]
                print "leading word is {}\n".format(leading_word)
                suffix_words = lst[1:][0]
                print "suffixing word is {}\n".format(suffix_words)
    #            for each in suffix_words:
    #                print each
                leading_score = self._score_by_len(leading_word)
                for each in suffix_words:
                    print "working on word: {}\n".format(each)
                    suffix_score, suffix_list = search(each)
                    yield (leading_score + suffix_score, [leading_word] + suffix_list)
                    

            return self._max(list(candidates()))



        return search(lst)


    def opt(self, component):
        optimalized_components = []
        if True:
            print "inside opt\n"
            if len(component.nodes()) == 1:
               #means the component contains only one word
               print "single compond: {}\n".format(component.nodes())
               return True


            #return a list of optimal words in format of (pos, "word")
    
            nodes = component.nodes() #initially nodes = all nodes in component
            nodes.sort()
            print "nodes are {}\n".format(nodes)

            def search(component, nodes=nodes, node=nodes[0], flag='init'):
                if not nx.non_neighbors(component, node) and flag != 'init':
                    print "no neighbor returned: {}".format([node])
                    return node
                elif nx.non_neighbors(component, node) and flag != 'init':
                        #only look at word forwad
                    flag = "HASNOT"
                    for each in nx.non_neighbors(component, node):

                        if each[0][0] > node[0][0]:
                            #if non_neighbor has forward neighbors
                            #keep the flag
                            flag = "HAS"
                            break
                        else:
                            #all non_neighbors are in front of the node
                            pass

                    if flag == "HASNOT":
                        print "no forward neighbor returned: {}".format([node])
                        return node
                    else:
                        #means it has non_neighbor following it
                        pass
                
                def candidates():
                    for node in nodes:
                            print "node is {}\n".format(node)
                            #node, say, ((0, 3),"face")

                            if list(nx.non_neighbors(component, node)) != []:
                             for each_non_neighbor in nx.non_neighbors(component, node):
                                candidate_nodes = [node]

                                if each_non_neighbor[0][0] > node[0][0]:
                                    print "each_non_neighbor is {}\n".format(each_non_neighbor)
                                #boo is one of the non_neighbors of "face"
                                    candidate_nodes.append(search(component, [each_non_neighbor], each_non_neighbor, flag=''))
                                #booking, no further non_neighbors, so will return a node itself
                                #boo, has two non_neighbors -- "king" and "girl", will return the 
                                yield candidate_nodes
                                #yield (score, [list of nodes])
                            else:
                                print "HHHHEEEERRRREEEE\n"
                                candidate_nodes = [node]
                                yield candidate_nodes

                return list(candidates())

            optimized_words = search(component) #in format of (pos, "word1xxword2..")
            print "optimized_words is {}\n".format(optimized_words)
            s = []
            for i in optimized_words:
                if i not in s:
                    s.append(i)
            print "s is {}".format(s)

            for each in s:
                print "\n"
                
                print each

        return s

    def segment2(self, component):
        pass

    #public interface II
    def segment(self, text):
        '''
        Public interface for user to find all meaningful words in a string.
        Input: string
        Return: a set of meaningful English words found
        '''
        if self._casesensitive == False:
            self._string = text.lower()
            self._string = self._string.strip("'")
        else:
            pass #for current version, only support lowercase version

#        temp_dic = dict()
        candidate_list = []
        pair_dic = dict()
        for prefix, suffix in self._divide():
            #yield ((self._string[counter:cut_point],(counter, counter+self._minlen)), self._string[cut_point:])
            #prefix is ("hello",(0,5))
            pair_dic[prefix] = suffix
            if prefix[0] in self.ngram_distribution:
                #temp_dic[prefix] = self.ngram_distribution[prefix]
                candidate_list.append(
                    (self.ngram_distribution[prefix[0]], prefix)
                )
            else:
                #means this prefix was not likely
                #to be a part of meaningful word
                pass

        candidate_list.sort(reverse=True)
        #now candidate list is [(2345234, ("hello",(0,5)))]
        meaningful_words = []
        #meaningful_words is [((0, 10),"helloworld"),...]
        for each in candidate_list:
            #(17507324569.0, ('in', (8, 10)))
            for word in self.ngram_tree[each[1][0]]:
                if word in each[1][0] + pair_dic[each[1]]:
                    if self._string[each[1][1][0]:each[1][1][0]+len(word)] == word:
                        meaningful_words.append(((each[1][1][0], each[1][1][0]+len(word)), word))
                    
        #sort the list in order of position in original text
        meaningful_words.sort()
        #the sorted list is [((0,5), "hello"),((0,10), "helloword"),...]
        components = []
        components = self._find_components(meaningful_words)
        print "debug: components are: {}\n".format(components)
        for each in components:
            temp_lst = each.nodes()
            temp_lst.sort()
            print "component: {}\n".format(temp_lst)
        

        post_components = []
        for each in components:
            print "debug: working on: {}\n".format(each.nodes())
            post_components.append(self._component_optimizing(each))
            #returns ((start_pos, end_pos), segment_word_lst)

        
        meaningful_pos_lst = []
        for each in post_components:
            print "post_component: {}\n".format(each)
            meaningful_pos_lst += range(each[0][0], each[0][1])

        non_meaning_pos_lst = []
        for x in xrange(len(self._string)):
            if x in meaningful_pos_lst:
                continue
            else:
                non_meaning_pos_lst.append(x)
                
        non_meaningful_range = []
        non_meaningful_range = [as_range(g) for _, g in groupby(non_meaning_pos_lst, key=lambda n, c=count(): n-next(c))]
        #[(8, 9), (12, 30)]

        print "non-meaningful list: {}\n".format(non_meaningful_range)
        meaningful_dic = dict()

        overall_pos_lst = []
        for each in non_meaningful_range:
            overall_pos_lst.append(each)
        for component in post_components:
            overall_pos_lst.append(component[0])
            meaningful_dic[component[0]] = component[1]

        overall_pos_lst.sort()
        print overall_pos_lst
        return_lst = []
        for each in overall_pos_lst:
            if each in meaningful_dic:
                return_lst.extend(meaningful_dic[each])
            else:
                return_lst.append(self._string[each[0]:each[1]+1])

        
        print "RESULT: {}\n".format(return_lst)
        

        return return_lst
        #[[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok")],[((8, 19),"anotherword")]]
#        optimized_components = []
#        optimized_components = _component_optimize(self, components)    
                
        #replace the optimzed component result to the original str
#        return set(meaningful_words)


__title__ = 'English Word Finder'
__version__ = '0.150'
__build__ = 0x0003
__author__ = 'Weihan Jiang'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright (c) 2015 Weihan Jiang'
