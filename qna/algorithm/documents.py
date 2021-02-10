import pandas as pd
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy import dot
from numpy.linalg import norm
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi
from eunjeon import Mecab  # KoNLPy style mecab wrapper



# 형태소 분석기 로드
def getTagger():
    return Mecab()

# 형태소 분석기를 이용하여 문장 토큰화
def tokenize(sent):
    tagger = getTagger()
    return tagger.morphs(sent)

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

def getTopicContent(docs):
    topicPlusContent = []

    for doc in docs:
        topicPlusContent.append(list(doc)[0] + ' ' + list(doc)[1])

    return topicPlusContent

def getNouns(query):
    tagger = getTagger()
    return tagger.nouns(query)

# 문서 검색 알고리즘
def search(query):
    topicPlusContent = getTopicContent(selectMainFromData())
    tokenized_corpus = [tokenize(doc) for doc in topicPlusContent]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = getNouns(query)
    return bm25.get_top_n(tokenized_query, topicPlusContent, n=1)[0]
