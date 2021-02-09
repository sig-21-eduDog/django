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
    conn = pymysql.connect(host='localhost', user='RLJG_MANAGER', password='q1W2e3R4', db='Data')
    SQL = 'SELECT MAIN FROM DATA'
    curs = conn.cusor()
    curs.execute(SQL)
    data = curs.fetchall()
    curs.close()
    conn.close()
    return data


# excel_data = pd.read_excel('naver_it2.xlsx', engine ='openpyxl')
def search(query):
    corpus = selectMainFromData()
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query)
    # doc_scores = bm25.get_scores(tokenized_query)
    return bm25.get_top_n(tokenized_query, corpus, n=1)
