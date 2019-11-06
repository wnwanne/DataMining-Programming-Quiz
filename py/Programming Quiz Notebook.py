#!/usr/bin/env python
# coding: utf-8

# # Task:
# The task is to estimate appropriate parameters using the training data, and use it to predict reviews from the test data, and classify each of them as either positive or negative.

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import string
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import math
from matplotlib import rc
#from google.colab import drive
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics  import accuracy_score
from nltk.tokenize import word_tokenize

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
nltk.download('stopwords')
nltk.download('punkt')


# In[ ]:


def preview_df(_df):
    print(_df.shape)
    return df.head()

class Tokenizer:
    
    def clean(self, text):
        no_html = BeautifulSoup(text).get_text()
        clean = re.sub("[^a-z\s]+", " ", no_html, flags=re.IGNORECASE)
        return re.sub("(\s+)", " ", clean)

    def tokenize(self, text):
        clean = self.clean(text).lower()
        stopwords_en = stopwords.words("english")
        return [w for w in re.split("\W+", clean) if not w in stopwords_en]

class MultinomialNaiveBayes:
    
    def __init__(self, classes, tokenizer):
        self.tokenizer = tokenizer
        self.classes = classes
    
    def group_by_class(self, X, y):
        data = dict()
        for c in self.classes:
            data[c] = X[np.where(y == c)]
        return data
           
    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        
        grouped_data = self.group_by_class(X, y)
        
        for c, data in grouped_data.items():
            self.n_class_items[c] = len(data)
            self.log_class_priors[c] = math.log(self.n_class_items[c] / n)
            self.word_counts[c] = defaultdict(lambda: 0)
          
            for text in data:
                counts = Counter(self.tokenizer.tokenize(text))
                for word, count in counts.items():
                    if word not in self.vocab:
                        self.vocab.add(word)

                    self.word_counts[c][word] += count
        return self
      
    def laplace_smoothing(self, word, text_class):
        num = self.word_counts[text_class][word] + 1
        denom = self.n_class_items[text_class] + len(self.vocab)
        return math.log(num / denom)
      
    def predict(self, X):
        result = []
        for text in X:
            
            class_scores = {c: self.log_class_priors[c] for c in self.classes}

            words = set(self.tokenizer.tokenize(text))
            for word in words:
                if word not in self.vocab: continue

                for c in self.classes:
                
                    log_w_given_c = self.laplace_smoothing(word, c)
                    class_scores[c] += log_w_given_c
                
            result.append(max(class_scores, key=class_scores.get))

        return result


# In[ ]:


path = 'C:\\Users\\Owner\\Dev\\Python\\Data Mining\\ProgrammingQuizData\\aclImdb'


# In[ ]:


folder = 'C:\\Users\\nwannew\\Documents\\Dev\\School\\intro to data mining\\aclImdb'
labels = {'pos' : 1, 'neg': 0}
df = pd.DataFrame()


# In[ ]:


for f in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']
preview_df(df)


# In[ ]:


f = sns.countplot(x='sentiment', data=train_df)
f.set_title("Training Sentiment distribution")
f.set_xticklabels(['Negative', 'Positive'])
plt.xlabel("");


# In[ ]:


f = sns.countplot(x='sentiment', data=test_df)
f.set_title("Testing Sentiment distribution")
f.set_xticklabels(['Negative', 'Positive'])
plt.xlabel("");


# So the first task is to go through all the files in the ‘train’ folder, and construct the vocabulary V of all unique words. Please ignore all the stop-words. The words from each file (both in training and testing phase) must be extracted by splitting the raw text only with whitespace characters and converting them to lowercase characters. 

# In[ ]:


reviews = df.review.str.cat(sep=' ')
#function to split text into word
tokens = tokenize(reviews)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]


# In[ ]:


stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]


# In[ ]:


wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# ## Building the classifier

# In[ ]:


X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# In[ ]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)


# In[ ]:


alpha = 1.0

clf = MultinomialNB(alpha=alpha,fit_prior=True).fit(train_vectors, y_train)


# In[ ]:


predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))


# In[ ]:


test_path = os.path.join(path2, 'test')
train_path = os.path.join(path2, 'train')
pos_test = os.path.join(test_path, 'pos')
neg_test = os.path.join(test_path, 'neg')
pos_train = os.path.join(train_path, 'pos')
neg_train = os.path.join(train_path, 'neg')


# In[ ]:


pos_train_list = []
for file in os.listdir(pos_train):
    file_path = os.path.join(pos_train, file)
    with open(file_path, 'r', encoding='utf8') as n:
        line = n.read()
        pos_train_list.extend(tokenize(line))


# In[ ]:


neg_train_list = []
for file in os.listdir(pos_train):
    file_path = os.path.join(pos_train, file)
    with open(file_path, 'r', encoding='utf8') as n:
        line = n.read()
        neg_train_list.extend(tokenize(line))


# In[ ]:


pos_test_list = []
for file in os.listdir(pos_train):
    file_path = os.path.join(pos_train, file)
    with open(file_path, 'r', encoding='utf8') as n:
        line = n.read()
        pos_test_list.extend(tokenize(line))


# In[ ]:


neg_test_list = []
for file in os.listdir(pos_train):
    file_path = os.path.join(pos_train, file)
    with open(file_path, 'r', encoding='utf8') as n:
        line = n.read()
        neg_test_list.extend(tokenize(line))


# In[ ]:


vocab_list = list(itertools.chain(pos_test_list,neg_test_list,pos_train_list,neg_train_list))
unique_vocab_list = set(vocab_list)


# In[ ]:


unique_list = set(vocab_list)
len(unique_list)


# The next step is to get counts of each individual words for the positive and the negative classes
# separately, to get P(word|class).

# In[ ]:


#counts per values
pos_dict_counts = Counter(pos_train_list)


# In[ ]:


#counts per value
neg_dict_counts = Counter(neg_train_list)


# Finding the log-posterior (un-normalized), which is given by log(P(X|Y )P(Y )), for both the classes with laplace smoothing.

# In[ ]:


pos_logs = {}
for c, data in pos_dict_counts.items():
    logs[c] = math.log(data + 1 / (len(pos_train_list) + 1) +
                       len(pos_train_list)/(len(pos_train_list)+len(neg_train_list)))
neg_logs = {}
for c, data in neg_dict_counts.items():
    logs[c] = math.log(data + 1 / (len(neg_train_list) + 1) + 
                       len(neg_train_list)/(len(pos_train_list)+len(neg_train_list)))


# In[ ]:




