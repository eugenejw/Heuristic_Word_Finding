import sys
from os.path import join, dirname, realpath
from math import log10

def parse_file(filename):
    """Read `filename` and parse tab-separated file of (word, count) pairs."""
    with open(filename) as fptr:
        lines = (line.split('\t') for line in fptr)
        return dict((word, float(number)) for word, number in lines)

unigram_counts = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))


lst = [((0, 4), 'face'), [[((4, 6), 'bo'), ((10, 14), 'girl')], [((4, 6), 'bo')], [((4, 6), 'bo')], [((4, 6), 'bo'), ((8, 11), 'king')], [((4, 6), 'bo'), [[((6, 9), 'ok')], [((6, 9), 'ok'), ((10, 14), 'girl')], [((6, 9), 'ok')], [((6, 9), 'ok')]]]]]
lst = [((0, 4), 'face'), [[((4, 8), 'book'), [[((8, 10), 'in')], [((8, 10), 'in')], [((8, 10), 'in'), ((10, 14), 'girl')], [((8, 10), 'in')], [((8, 10), 'in')], [((8, 10), 'in')], [((8, 10), 'in')]]], [((4, 8), 'book'), ((10, 14), 'girl')], [((4, 8), 'book')], [((4, 8), 'book')], [((4, 8), 'book'), ((8, 11), 'king')]]]

def _penalize(self, lst, s=0, e=14):
        #(0.00011509110207386408, [((1, 3), 'ac'), ((7, 11), 'king')])
        #(-8.483260969067402, [((1, 3), 'ac'), ((7, 11), 'king'), ((11, 13), 'ir')])
        if not lst[1][0][0][0] - start_pos == 0:
            starting_pos_penalty = log10(10.0 / (1024908267229.0 * 10 ** (lst[1][0][0][0] - start_pos)))
        else:
            starting_pos_penalty = 0

        if not end_pos - lst[1][-1][0][1] == 0:
            ending_pos_penalty = log10(10.0 / (1024908267229.0 * 10 ** (end_pos - lst[1][-1][0][1])))
        else:
            ending_pos_penalty = 0

        interval_penalty = [0]
        count = len(lst[1])
        for i in xrange(count-1):
            if not lst[1][i+1][0][0] - lst[1][i][0][1] == 0:
                interval_penalty.append(log10(10.0 / (1024908267229.0 * 10 ** (lst[1][i+1][0][0] - lst[1][i][0][1]))))
        return sum(interval_penalty) + starting_pos_penalty + ending_pos_penalty
            
            


def score_by_len(lst):
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

def score1(lst):
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



def _max(lst):
    print "debug: input list is {}".format(lst)
    tmp_lst = []
    for each in lst:
        tmp_lst.append(each[0])

    max_score = max(tmp_lst)
    
    winners = []
    for each in lst:
        if each[0] == max_score:
            
            winners.append((score1(each[1]), each[0], each[1]))

    print winners
    print "winner is {}".format(max(winners))
    return [max(winners)[1], max(winners)[2]]


def segment(lst):
    """return a list of words that is the best segmentation of 'text'"""
    
    def search(lst):

        flag = "ALL_NON_LIST"
        if len(lst) == 1 and not isinstance(lst[0], list):
            print "returned {}\n".format(lst)
            
            return (score_by_len(lst), lst)
        for each in lst:
            if isinstance(each, list):
                flag = "LIST_FOUND"
        if flag == "ALL_NON_LIST":
            print "returned {}\n".format(lst)
            return (score_by_len(lst), lst)

        def candidates():
            leading_word = lst[0]
            print "leading word is {}\n".format(leading_word)
            suffix_words = lst[1:][0]
            print "suffixing word is {}\n".format(suffix_words)
#            for each in suffix_words:
#                print each
            leading_score = score_by_len(leading_word)
            for each in suffix_words:
                print "working on word: {}\n".format(each)
                suffix_score, suffix_list = search(each)
                print "{}".format((leading_score + suffix_score, [leading_word] + suffix_list))
                yield (leading_score + suffix_score, [leading_word] + suffix_list)

        return _max(list(candidates()))
            


    return search(lst)
    
print segment(lst)
'''
result_lst = []
print "input_lst is {}\n".format(lst[1:][0])
result_lst = print_tree(lst[1:][0])
for each in  result_lst[0]:
    print each
'''
