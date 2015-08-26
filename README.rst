English Word Finding in Python
------------------------------------
        
WordFinder is a module finding English word(s) from a string in a heuristic manner.
Written in pure-Python, and based on a trillion-word corpus.

Typically, the problem is to find whether the following string contains meaningful word whose len(word)>= number (ie, 5).
"fsdkjqlerjgwejrgjeqoigrghnjksnvasnva^^safq*wjrfdgivjqergjqeuniversitylqpweovar'qemrbvqebebq".

Maybe the hardest part of find meaningful words is to do the job wisely rather than costing O(n^2).

In this WordFinder module, a heuristic way has been adopted to reduce the complexity from O(n^2) to O(n).

The heuristic flavor is built on top of  enchant English dictionary together with Google unigram word corpus.

Using the corpus was inspired by the chapter "Natural Language Corpus Data" by Peter Norvig,
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009). Data files are derived from the Google Web Trillion Word Corpus, as described by Thorsten Brants and Alex Franz, and distributed by the Linguistic Data Consortium. 

This module contains only a subset of that data. The unigram data includes only the most common 333,000 words. Every word and phrase is lowercased with punctuation removed.

I've further truncated the corpus by removing all non-dictionary words(words not found in enchant English Dic). The dictionary I used to truncate it, is Python Enchant Dictionary.
So any word found in the truncated corpus will be deemed as meaningful.

Natural Language Corpus Data:  `Natural Language Corpus Data
<http://norvig.com/ngrams/>`_.

Beautiful Data: `Beautiful Data
<http://oreilly.com/catalog/9780596157111/>`_.

Google Web Trillion Word Corpus: `Google Web Trillion Word Corpus <http://googleresearch.blogspot.com/2006/08/all-our-n-gram-are-belong-to-you.html>`_.

Distributed: `distributed 
<https://catalog.ldc.upenn.edu/LDC2006T13>`_.

Python enchant: `Python enchant
<https://pypi.python.org/pypi/pyenchant/>`_.

Features
----------

- Pure-Python
- Fully documented
- Includes unigram data
- Command line interface for batch processing
- Easy to hack (e.g. new data, different language)
- Developed on Python 2.7
- Tested on CPython 2.7 and 3.4


User Guide
-------------

Installing wordfinder is simple with pip

::

  >>> pip install wordfinder

In your own Python programs, you'll mostly want to use *search* to query whether a string contains any meaningful words whose length >= N(set by user). True is returned if found.

::

    >>> from wordfinder import WordFinder
    #instantiate and set the minimal word length(in this example, 5)
    >>> wf_5 = WordFinder(5)
    #start to search whether word with len(word)>=5 is in the given string.
    >>> wf_5.search("first-sring--sfasggregqgqvrjeykjwj")
    True


Instantiation is no longer needed for searches with the same criteria "len(word)>=5",

::

    >>> wf_5.search("second-sring--iyejhwrjqihrgq;nergqrgjqiergqebef")

But you need to create another instance for a different criteria, say "len(word)>=6",

::

    >>> wf_6 = WordFinder(6)
    >>> wf_6.search("third-string--jfqsdfauniversityregqwvqedfqrewgfqgweijfq")
    True

Additionally, you could use *get* to get all meaningful words from a string,

::

    >>> wf_5.get("asdsdgfuierhghelloafsdjkasjdf@#$#sdfsuniversityadfsaof*washington")
    ['university', 'washing', 'washington', 'hello', 'ashing']


API Documentation
-----------------

- search(text)

    Return True, if text contains meaningful word of minimal length.
    Return False, if no meaningful word is found from the string.

- get(text)

    Return a list of meaningful words(matches the minimal length you've set) found from a string.

Useful Links
------------

Heuristic_Word_Finding @ Github, `Heuristic_Word_Finding
<https://github.com/eugenejw/Heuristic_Word_Finding>`_.

Report bugs, `issue tracker
<https://github.com/eugenejw/Heuristic_Word_Finding/issues>`_.


License
---------
        
        Copyright (c) 2015 Weihan Jiang
        
           Licensed under the Apache License, Version 2.0 (the "License");
           you may not use this file except in compliance with the License.
           You may obtain a copy of the License at
        
               http://www.apache.org/licenses/LICENSE-2.0
        
           Unless required by applicable law or agreed to in writing, software
           distributed under the License is distributed on an "AS IS" BASIS,
           WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
           See the License for the specific language governing permissions and
           limitations under the License.
        
Keywords: word search,word find,word searching,meaningful words,from string,from text
Platform: any
Classifier: Development Status :: 4 - Beta
Classifier: Intended Audience :: Developers
Classifier: Natural Language :: English
Classifier: License :: OSI Approved :: Apache Software License
Classifier: Programming Language :: Python
Classifier: Programming Language :: Python :: 2.7
Classifier: Programming Language :: Python :: 3.4
