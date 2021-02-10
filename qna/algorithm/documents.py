import pandas as pd
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy import dot
from numpy.linalg import norm
import numpy as np
from rank_bm25 import BM25Okapi
import pymysql


def tokenizer(sent):
    return sent.split(" ")


def selectMainFromData():
    conn = pymysql.connect(host='localhost', user='RLJG_MANAGER', password='q1W2e3R4', db='rljg_schema')
    SQL = 'SELECT MAIN FROM DATA'
    curs = conn.cursor()
    curs.execute(SQL)
    data = curs.fetchall()
    data = list(data)
    curs.close()
    conn.close()
    return data


def makelist(data):
    contentslist = []
    for i in range(len(data)):
        contentslist.append(list(data[i])[0])

    print(type(contentslist[10000]), contentslist[10000])
    return contentslist


def search(query):
    corpus = makelist(selectMainFromData())
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query)
    return bm25.get_top_n(tokenized_query, corpus, n=1)[0]
