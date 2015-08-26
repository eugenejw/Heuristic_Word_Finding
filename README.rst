Heuristic English Word Finding in Python
===================================

WordFinder is a module finding English word(s) from a string in a heuristic manner.
Written in pure-Python, and based on a trillion-word corpus.

Typically, the problem is to find whther the following string contains meaningful word that has len(word)>= 5.
"fsdkjqlerjgwejrgjeqoigrghnjksnvasnva^^safq*wjrfdgivjqergjqeuniversitylqpweovar'qemrbvqebebq"

Maybe the hardest part of find meaningful words is to do the job wisely rather than costing O(n^2).

In this WordFinder module, a heuristic way has been adopted to reduce the complexity from O(n^2) to O(n).

The heuristic flavor is built on top of  enchant English dictionary together with Google unigram word corpus.

Using the corpus was inspired by the chapter "`Natural Language Corpus Data`_" by Peter Norvig,
from the book "`Beautiful Data`_" (Segaran and Hammerbacher, 2009).
Data files are derived from the `Google Web Trillion Word Corpus`_, as described
by Thorsten Brants and Alex Franz, and `distributed`_ by the Linguistic Data
Consortium. This module contains only a subset of that data. The unigram data
includes only the most common 333,000 words. 
Every word and phrase is lowercased with punctuation removed.

I've further truncated the corpus by removing all non-dictionary words(words not found in enchant English Dic).
The dictionary I used to truncate it, is Python Enchant Dictionary.
So any word found in the truncated corpus will be deemed as meaningful.

.. _`Natural Language Corpus Data`: http://norvig.com/ngrams/
.. _`Beautiful Data`: http://oreilly.com/catalog/9780596157111/
.. _`Google Web Trillion Word Corpus`: http://googleresearch.blogspot.com/2006/08/all-our-n-gram-are-belong-to-you.html
.. _`distributed`: https://catalog.ldc.upenn.edu/LDC2006T13
.. _`Python enchant`: https://pypi.python.org/pypi/pyenchant/

Features
--------

- Pure-Python
- Fully documented
- Includes unigram data
- Command line interface for batch processing
- Easy to hack (e.g. new data, different language)
- Developed on Python 2.7
- Tested on CPython 2.7 and 3.4

User Guide
----------

Installing wordfinder is simple with pip

    >>> pip install wordfinder

In your own Python programs, you'll mostly want to use *search* to query
whether a string contains any meaningful words whose length >= N(set by user).
True is returned if found.


    >>> from wordfinder import WordFinder
    #instantiation. At the instantiating time, the minimal word length is set.
    >>> wf = WordFinder(5)
    #start to search whether word with len(word)>=5 is in the given string.
    >>> wf.search("afqwerfqvqervqtrehghqehelloworldasfsvsdv sfaqsdf")
    >>> True

Additonally, you could use *get* to get all meaningful words from a string.

    >>> wf.get("asdsdgfuierhghelloafsdjkasjdf@#$#sdfsuniversityadfsaof*washington")
    >>> ['university', 'washing', 'washington', 'hello', 'ashing']


API Documentation
-----------------

- search(text)

    Return True, if text contains meaningful word of minimal length.
    Return False, if no meaningful word is found from the string.

- get(text)

    Return a list of meaningful words(matches the minimal length you've set) found from a string.


Useful Links
------------

- `Heuristic_Word_Finder @ Github`_
- `Issue Tracker`_

.. _`Heuristic_Word_Finding @ Github`: https://github.com/eugenejw/Heuristic_Word_Finding
.. _`Heuristic_Word_Finding @ Github`: https://github.com/eugenejw/Heuristic_Word_Finding/issues


License
-------

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
