#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)  # drqa/retriever/__init__.py --> doc_db.py
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()  # Fetch all ids of docs stored in the db.
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}  # store in {'3255': 0, '8902': 1, ...}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)  # 'corenlp', drqa/tokenizers/__init__.py --> corenlp_tokenizer.py
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]  # total 10 batches
    _count = partial(count, args.ngram, args.hash_size)  # args.hash_size --> default=int(math.pow(2, 24))
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    """
    csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
    
    Examples:
        >>> row = np.array([0, 0, 1, 2, 2, 2])
        >>> col = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
        array([[1, 0, 2],
               [0, 0, 3],
               [4, 5, 6]])
    
    count_matrix: shape=(args.hash_size, len(doc_ids))
    
              doc_1   doc_2  ...   doc_m
    word_1    [[1,      0,   ...    2],
    word_2     [0,      0,   ...    3],
     ...                ...
    word_n     [4,      5,   ...    6]]
    
    i.e., (word_1, doc_m) denotes word 'word_1' appear 2 times in doc 'doc_m'.
    
    Reference: https://towardsdatascience.com/machine-learning-to-big-data-scaling-inverted-indexing-with-solr-ba5b48833fb4
    """
    count_matrix = sp.csr_matrix(  # import scipy.sparse as sp
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    """
    >>> idfs
    array([-0.51082562,  0.51082562, -1.94591015])
    >>> idfs[idfs < 0]=0
    >>> idfs
    array([0.        , 0.51082562, 0.        ])
    >>> idfs = sp.diags(idfs, 0)
    >>> idfs.toarray()
    array([[0.        , 0.        , 0.        ],
           [0.        , 0.51082562, 0.        ],
           [0.        , 0.        , 0.        ]])
    >>> aa.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]], dtype=int64)
    >>> tfs = aa.log1p()
    >>> tfs.toarray()
    array([[0.69314718, 0.        , 1.09861229],
           [0.        , 0.        , 1.38629436],
           [1.60943791, 1.79175947, 1.94591015]])
    >>> tfidfs = idfs.dot(tfs)
    >>> tfidfs.toarray()
    array([[0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.70815468],
           [0.        , 0.        , 0.        ]])
    """
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()  # Returns a new tensor with the natural log of (1 + x_i).
    """
    (args.hash_size, args.hash_size) .doc (args.hash_size, len(doc_ids)) = (args.hash_size, len(doc_ids))
    """
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""

    """Examples:
    >>> aa=sp.csr_matrix((data, (row, col)), shape=(3, 3))
    >>> aa
    <3x3 sparse matrix of type '<class 'numpy.int64'>'
            with 6 stored elements in Compressed Sparse Row format>
    >>> aa.sum_duplicates()
    >>> aa
    <3x3 sparse matrix of type '<class 'numpy.int64'>'
            with 6 stored elements in Compressed Sparse Row format>
    >>> aa.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]], dtype=int64)
    >>> (aa > 0).astype(int).toarray()
    array([[1, 0, 1],
           [0, 0, 1],
           [1, 1, 1]])
    >>> np.array((aa > 0).astype(int).sum(1))  # each row denotes each word.
    array([[2],
           [1],
           [3]])
    >>> np.array((aa > 0).astype(int).sum(1)).shape
    (3, 1)
    >>> np.array((aa > 0).astype(int).sum(1)).squeeze()
    array([2, 1, 3])
    >>> np.array((aa > 0).astype(int).sum(1)).squeeze().shape
    (3,)
    """
    binary = (cnts > 0).astype(int)
    """
    freqs: In all docs, the nums of docs which appear 'word_n'.
            word_1  word_2  ...  word_n
        array([2,     1,    ...,   3])
    """
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    # 1.Form a sparse word to document count matrix (inverted index).
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    # 2.Convert the word count matrix into tfidf one.
    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)  # the matrix with each element is a tfidf value.

    # 3.Return word --> # of docs it appears in.
    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    # define filename
    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))  # (default=2, default=int(math.pow(2, 24)), 'corenlp')
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,  # for each word (or keyword), # of docs it appears in.
        'tokenizer': args.tokenizer,  # 'corenlp'
        'hash_size': args.hash_size,  # default=int(math.pow(2, 24))
        'ngram': args.ngram,  # default=2
        'doc_dict': doc_dict  # (DOC2IDX, doc_ids)
    }
    """
    drqa/retriever/__init__.py --> utils.py
    """
    retriever.utils.save_sparse_csr(filename, tfidf, metadata)
