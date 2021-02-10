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


# 공백 기준으로 자르기
def tokenizer(sent):
    return sent.split(" ")


# db에서 본문만 들고오기
def selectMainFromData():
    conn = pymysql.connect(host='localhost', user='RLJG_MANAGER', password='q1W2e3R4', db='rljg_schema')
    SQL = 'SELECT TITLE, MAIN FROM DATA'
    curs = conn.cursor()
    curs.execute(SQL)
    data = curs.fetchall()
    # data가 tuple의 tuple이므로 tuple의 list로 바꿔준다.
    data = list(data)
    curs.close()
    conn.close()
    return data


# selectMainFromData()의 반환값이 tuple의 list이므로, tuple을 또 list로 바꾸어 0번째 항목(main)을 별개의 list에 저장
def makelist(data):
    contentslist = []
    topicPlusContent = []
    for i in range(len(data)):
        # 문서 내용 저장
        contentslist.append(list(data[i])[1])
        # 본문에 주제가 없는 경우가 많아서 임의로 추가해 줌
        topicPlusContent.append(list(data[i])[0] + ' ' + list(data[i])[1])
    return contentslist, topicPlusContent


# 문서 검색 알고리즘
def search(query):
    contentslist, topicPlusContent = makelist(selectMainFromData())
    tokenized_corpus = [tokenizer(doc) for doc in topicPlusContent]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query)
    return bm25.get_top_n(tokenized_query, contentslist, n=1)[0]
