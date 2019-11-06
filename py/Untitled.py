#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import string
import re
import os
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
nltk.download('stopwords')


# # Helper Functions:

# In[ ]:


def preview_df(_df):
    print(_df.shape)
    return _df.head()

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
      
    def laplace_smoothing(self, word, text_class, alpha=1000):
      num = self.word_counts[text_class][word] + alpha
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


# # Load Data

# In[ ]:


folder = 'C:\\Users\\nwannew\\Documents\\Dev\\School\\intro to data mining\\aclImdb'
labels = {'pos' : 1, 'neg': 0}

train = pd.DataFrame()
test = pd.DataFrame()

train_folder = os.path.join(folder, 'train')
test_folder = os.path.join(folder, 'test')


# In[ ]:


for l in ('pos', 'neg'):
    path = os.path.join(train_folder, l)
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf8') as infile:
            txt = infile.read()
        train = train.append([[txt, labels[l]]], ignore_index=True)
train.columns = ['review', 'sentiment']
preview_df(train)


# In[ ]:


for l in ('pos', 'neg'):
    path = os.path.join(test_folder, l)
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf8') as infile:
            txt = infile.read()
        test = test.append([[txt, labels[l]]], ignore_index=True)
test.columns = ['review', 'sentiment']
preview_df(test)


# # Visuals

# In[ ]:


f = sns.countplot(x='sentiment', data=train)
f.set_title("Sentiment distribution")
f.set_xticklabels(['Negative', 'Positive'])
plt.xlabel("");


# In[ ]:


text = " ".join(review for review in train.review)


wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords.words("english")).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show();


# # Training:

# In[ ]:


X = train['review'].values
y = train['sentiment'].values
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[ ]:


MNB = MultinomialNaiveBayes(classes=np.unique(y), 
                            tokenizer=Tokenizer()).fit(X_train, y_train)


# # Prediction:

# In[ ]:


y_hat = MNB.predict(X_test)


# # Accuracy:

# In[ ]:


accuracy_score(y_test, y_hat)


# # Confusion Matrix:

# In[ ]:


print(classification_report(y_test, y_hat))


# In[ ]:


cnf_matrix = confusion_matrix(y_test, y_hat)
cnf_matrix


# In[ ]:


class_names = ["negative", "positive"]
fig,ax = plt.subplots()


sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.ylabel('Actual sentiment')
plt.xlabel('Predicted sentiment');


# Why do you think the accuracy suffers when α is too high or too low?:
# 
# The accuracy most likely suffers due to the scaling that takes place when alpha is either decreased 
