import pandas as pd
import re
import os
from nlputils import lexical
normalizer = lexical.Preprocessing()

class Preprocessing:
    def __init__(self, xlsx = '../data/tweets.xlsx'):
        self.path = xlsx
        #carrega o corpus
        self.dataframe = pd.read_excel(xlsx)
        self.dataframe = self.dataframe.fillna('empty')
    
    def create_dataframe(self):
        corpora = pd.read_excel(self.path)
        del corpora['Unnamed: 10']
        del corpora['Unnamed: 11']
        del corpora['Unnamed: 12']
        return corpora
        
    def remove_username(self, tweet):
        text = re.sub('@[^\s]+','',tweet)
        return text

    def remove_end_of_line(self, tweet):
        return tweet.replace('\n', '').replace('RT', '')

    def remove_duplicate_letters(self, tweet):
        # ([^rs])  - Qualquer letra que não r ou s
        # (?=\1+)  - Que se repita uma vez ou mais
        # |(rr)    - Ou dois r's
        # (?=r+)   - Que tenham mais r's à frente
        # |(ss)    - Ou dois s's
        # (?=s+)   - Que tenham mais s's à frente
        regex = r"([^rs])(?=\1+)|(rr)(?=r+)|(ss)(?=s+)"
        tweet = re.sub(regex, '', tweet, 0)
        return tweet
    
    def remove_url(self, tweet):
        return re.sub(r'https.*', '', tweet, flags=re.MULTILINE)
    
    def normalize(self, tweet):
        #print(tweet)
        tweet = self.remove_username(tweet)
        #print('username:\n', tweet)
        tweet = self.remove_url(tweet)
        #print('url:\n', tweet)
        tweet = self.remove_end_of_line(tweet)
        #print('endl:\n', tweet)
        tweet = self.remove_duplicate_letters(tweet)
        #print('duppl:\n', tweet)
        tweet = normalizer.tokenize_sentences(tweet)
        #print('sentences:\n', tweet)
        tweet = normalizer.lowercase(tweet)
        #print('lower:\n', tweet)
        tweet = normalizer.remove_punctuation(tweet)
        #print('punct:\n', tweet)
        tweet = normalizer.tokenize_words(tweet)
        #print('words:\n', tweet)
        return tweet

    def split(self, probability=0.8):
        #divide os conjuntos de treinamento e de teste
        if probability < np.random.rand():
            return 'train'
        return 'test'
    
    def sett(self):
        my_set = []
        for i in range(0, len(self.dataframe['text'])):
            my_set.append(self.split())
        self.dataframe['set'] = my_set
        self.dataframe['tag'] = self.dataframe['normalize'].apply(self.tag)
