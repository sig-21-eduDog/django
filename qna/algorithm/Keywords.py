from krwordrank.word import KRWordRank
from krwordrank.word import summarize_with_keywords
from gensim.summarization.summarizer import summarize


# stopwords = ['그리고', '그래서']

def find(text):
    # texts = text.split(".")
    # keywords = summarize_with_keywords(texts, min_count=5, max_length=10, beta=0.85, max_iter=10, stopwords=stopwords, verbose=True)
    # keywords = summarize_with_keywords(texts) # with default arguments
    # keywords
    summarize(text)
    keywords = summarize_with_keywords(summarize(text).split("\n"))
    return keywords
