import pandas as pd
import re
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy import dot
from numpy.linalg import norm
import numpy as np
from rank_bm25 import BM25Okapi


def tokenizer(sent):
    return sent.split(" ")


# excel_data = pd.read_excel('naver_it2.xlsx', engine ='openpyxl')
# corpus = excel_data['main']


def search(query):
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer(query)
    # doc_scores = bm25.get_scores(tokenized_query)
    return bm25.get_top_n(tokenized_query, corpus, n=1)
