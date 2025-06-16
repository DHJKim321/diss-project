import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def reddit_tokenizer(text):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True).tokenize
    stop_words = set(stopwords.words("english"))
    tokens = tokenizer(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens
