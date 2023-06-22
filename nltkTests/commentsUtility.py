import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *


COMMENTS_FILE = "D:\Downloads\PF\Pandora dataset\\big5_comments.csv"


class CommentsUtility:

    def __init__(self):
        comments = pd.read_csv(COMMENTS_FILE, on_bad_lines='skip')
        self.df_commments = comments.loc[:, ['author', 'body', 'subreddit', 'lang']]


    def print(self):
        print(self.df_commments.columns)
        print(self.df_commments.head(100))


    def get_comments_df(self):
        return self.df_commments


    def test_tokenize(self):
        tknzr = TweetTokenizer()
        # Tokenize and print first 10 comments
        for i, comment in enumerate(self.df_commments['body']):
            if i < 10:
                print(i)
                print(tknzr.tokenize(comment))
                print("\n")
            else:
                break

    
    def test_stem(self):
        tknzr = TweetTokenizer()
        stemmer = PorterStemmer()
        # Stem and print first 10 comments
        for i, comment in enumerate(self.df_commments['body']):
            if i < 10:
                print(i)
                tokenized = tknzr.tokenize(comment)
                stemmed = [stemmer.stem(token) for token in tokenized]
                print(tokenized)
                print(stemmed)
                print("\n")
            else:
                break
