# -*- coding: utf-8 -*-

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

import wordsegment

class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import tox
        errno = tox.cmdline(self.test_args)
        sys.exit(errno)

with open('README.rst') as fptr:
    readme = fptr.read()

with open('LICENSE') as fptr:
    license = fptr.read()

setup(
    name='wordfinder',
    version=heuristicwordfinder.__version__,
    description='Find English word from string.',
    long_description=readme,
    author='Weihan Jiang',
    author_email='weihan.github@gmail.com',
    url='https://github.com/eugenejw/Heuristic_Word_Finding',
    license=license,
    py_modules=['wordfind'],
    packages=['wordsegment_data'],
    package_data={'wordsegment_data': ['*.txt']},
    tests_require=['tox'],
    cmdclass={'test': Tox},
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
)