import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("D:\ML\inc_data.csv", sep = ',', header = None,encoding='latin-1')
#meta.head()
# rename columns
meta.columns = ["incident_num","description","short_description","assignment_group","resolution",6,7,8,9,10,11,12,13,"service",15,16,17,18]
#meta.head()

inc = pd.DataFrame(meta)

incident=inc[['incident_num','description','assignment_group','resolution','service']]
incident.head()
incident["resolution"].fillna(value="No Resolution", inplace = True)
incident["service"].fillna(value="No service found", inplace = True)
assigns = [] 

# extract genres
for i in incident['service']: 
  assigns.append(list(i.split(",")))


#print(assigns)
all_assign = sum(assigns,[])
len(set(all_assign))

all_assign = nltk.FreqDist(all_assign) 

# create dataframe
all_assign_df = pd.DataFrame({'assigns': list(all_assign.keys()), 
                              'Count': list(all_assign.values())})
all_assign_df.head()
g = all_assign_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "assigns") 
ax.set(ylabel = 'Count') 
#plt.show()
# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text
pd.options.mode.chained_assignment = None 
incident['description']=incident['description'].apply(str)
incident['clean_desc'] = incident['description'].apply(lambda x: clean_text(x))

def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
 # plt.show()
  
# print 100 most frequent words 
freq_words(incident['clean_desc'], 100)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

incident['clean_desc'] = incident['clean_desc'].apply(lambda x: remove_stopwords(x))
from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(incident['service'])

# transform target variable
y = multilabel_binarizer.transform(incident['service'])
#print (multilabel_binarizer)
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)
# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(incident['clean_desc'], incident['service'], test_size=0.05, random_state=1)
# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

#vec_file = 'vectorizer.pickle'
#pickle.dump(tfidf_vectorizer, open(vec_file, 'wb'))

from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
lr = LogisticRegression(solver='lbfgs', max_iter=10000)
clf = OneVsRestClassifier(lr)
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

filename = 'finalized_model_service.sav'
pickle.dump(clf, open(filename, 'wb'))

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)
#y_pred[3]
# evaluate performance
f1_score(yval, y_pred, average="micro")
# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)
# Saving model to disk
