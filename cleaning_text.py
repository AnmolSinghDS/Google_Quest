import pandas as pd
import numpy as np

import nltk
import re
import string

from nltk.corpus import stopwords
stopword = stopwords.words('english')
wn = nltk.WordNetLemmatizer()



train = pd.read_csv('C:/Users/anmol/Desktop/ML_Project/Dataset/Google_Quest/train.csv')
print(train.head(10))

# Checking the shape of data
print('Training data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# Checking the summary of data
def summary(data):
    df = pd.DataFrame(data.dtypes, columns=['dtypes'])
    df = df.reset_index()
    df['Name'] = df['index']
    df.drop(['index'], axis=1, inplace=True)
    df = df[['Name', 'dtypes']]
    df['Missing'] = data.isnull().sum().values
    df['Unique'] = data.nunique().values
    return df

print(summary(train))

# CLeaning text by removing punctuation and extracting stopwords.
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopword]
    return text

train['question_title'] = train['question_title'].apply(lambda x: clean_text(x))
train['question_body'] = train['question_body'].apply(lambda x: clean_text(x))
train['answer'] = train['answer'].apply(lambda x: clean_text(x))
print(train.head())
