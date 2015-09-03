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
        
    def _group(self, meaningful_words):
        #[((0,5), "hello"),((0,10), "helloword"),...]
        for each in meaningful_words:
            
        pos_list = 
        for k, g in groupby(enumerate(data), lambda (i,x):i-x):
            print map(itemgetter(1), g)

    def _connected_components(neighbors):
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
        start_pos = node_list[0][0]  #0
        end_pos = node_list[-1][0]  #6
        original_string = self._string[start_pos:end_pos]
        
        for each in node_list:
            _intersect(self, original_string, each):
            
        if prev is None:
        if word in unigram_counts:
            # Probability of the given word.                                                                                                                                                                               
            return unigram_counts[word] / 1024908267229.0
        else:
            # Penalize words not found in the unigrams according                                                                                                                                                                       
            # to their length, a crucial heuristic.                                                                                                                                                                                    

            return 10.0 / (1024908267229.0 * 10 ** len(word))
    else:
        bigram = '{0} {1}'.format(prev, word)

        if bigram in bigram_counts and prev in unigram_counts:

            # Conditional probability of the word given the previous                                                                                                                                                                   
            # word. The technical name is *stupid backoff* and it's                                                                                                                                                                    
            # not a probability distribution but it works well in                                                                                                                                                                      
            # practice.                                                                                                                                                                                                                

            return bigram_counts[bigram] / 1024908267229.0 / score(prev)
        else:
            # Fall back to using the unigram probability.                                                                                                                                                                              

            return score(word)
        pass

    def _component_optimalize(self, components):
        
        optimalized_components = []
        for component in components:
            if len(component.nodes()) == 1:
               #means the component contains only one word
               optimalized_components.append((component.nodes()[0][0][0], component.nodes()[0][0][1])) #("pos, "word")
               continue

            def get_optimal_words(self, component):
                '''
                return a list of optimal words in format of (pos, "word")
                '''
                nodes = component.nodes() #initially nodes = all nodes in component

                def search(component, nodes, node='init'):
                    if not nx.non_neighbors(component, node) and node != 'init':
                        return _score([node])
                    elif nx.non_neighbors(component, node) and node != 'init':
                        #only look at word forwad
                        flag = True
                        for each in nx.non_neighbors(component, node):
                            if each[0][0] > node[0][0]:
                                pass
                            else:
                                flag = False
                        if flag = True:
                            return _score([node])
#                        return (component.nodes()[0][0][0], component.nodes()[0][0][1]) #("pos, "word")
                
                    def candidates():
                        for node in nodes.sort():
                            #node, say, ((0, 3),"face")
                            candidate_nodes = _score([node])
                            for each_non_neighbor in non_neighbors(component, node):

                                #boo is one of the non_neighbors of "face"
                                candidate_nodes.append(search(component, each_non_neighbor, each_non_neighbor))
                                #booking, no further non_neighbors, so will return a node itself
                                #boo, has two non_neighbors -- "king" and "girl", will return the 
                                
                                yield sscore(candidate_nodes)
                                #yield (score, [list of nodes])

                    return max(candidates())

                optimized_words = search(component, [],'init') #in format of (pos, "word1xxword2..")
                return optimized_words

            optimalized_components.append(get_optimal_words(self,component))
        return optimalized_components    


    def _find_components(self, meaningful_words):
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        G = nx.Graph()
        G = _init_graph(self, meaningful_words)
        components = []
        components = list(nx.connected_component_subgraphs(G)
        return components

    #public interface II
    def segment(self, text):
        '''
        Public interface for user to find all meaningful words in a string.
        Input: string
        Return: a set of meaningful English words found
        '''
        if self._casesensitive == False:
            self._string = text.lower()
        else:
            pass #for current version, only support lowercase version

#        temp_dic = dict()
        candidate_list = []
        pair_dic = dict()
        for prefix, suffix in self._divide():
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
            for word in self.ngram_tree[each[1][0]]:
                if word in each[1][0] + pair_dic[each[1][0]]:
                    meaningful_words.append((each[1][1][0], each[1][1][0]+len(word)), word)
                    
        #sort the list in order of position in original text
        meaningful_words.sort()
        #the sorted list is [((0,5), "hello"),((0,10), "helloword"),...]
        components = _find_components(meaningful_words)
        #[[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok")],[((8, 19),"anotherword")]]
        optimized_components = []
        optimized_components = _component_optimize(self, components)    
                
        #replace the optimzed component result to the original str
        return set(meaningful_words)

                         
    def _find_candidate(self, component):
        temp_lst = []
        candidate_lst = []
        for each in component:
            temp_lst.append(each[0][1])
        original_word = self._string[min(component):max(temp_lst)]

        top_down_candidates = []
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok")]
        #face boo
        #face book
        #face ok
        #facebook
        #ace boo
        #ace book
        #ace ok
        #boo
        #book
        #ok
        for each in component:
            non_intersect_lst = []
            for each_2 in component:
                if self._intersect(each[0], each_2[0]):
                    #if the last element of templst intersects with each2{0}
                    pass
                else:
                    #no intersect
                    non_intersect_lst.append(each_2)
            #non_intersect_lst = [((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok")]
            for x in non_intersect_lst:
                if self._intersect(each[0],x[0]):
                    
                    


            each_length = each[0][1] - each[0][0] + 1
            if each_length == len(original_word):
                top_down_candidates.append(each)
            
            preffix = []
            suffix = []
            auto_fill_prefix = ''
            for temp_each in component:
                if temp_each[0][1]<each[0][0]:
                    preffix.append(temp_each)
                elif temp_each[0][0]>each[0][1]:
                    suffix.append(temp_each)
            if len(preffix)==0:
                #means needs to fill with non-meaningful characters
                auto_fill_prefix = original_word[0:each[0][0]]
            else:
                
            








            
            


__title__ = 'English Word Finder'
__version__ = '0.150'
__build__ = 0x0003
__author__ = 'Weihan Jiang'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright (c) 2015 Weihan Jiang'
