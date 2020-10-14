import pandas as pd
import numpy as np
import nltk
import re
import heapq

train = pd.read_csv('C:/Users/anmol/Desktop/ML_Project/Dataset/Google_Quest/train.csv')

Quest_body = train['question_body'].values
print(Quest_body[0])

Cleaned_body = []
for i in range(Quest_body.shape[0]):
    sent = nltk.sent_tokenize(Quest_body[i])
    for c in range(len(sent)):
        sent[c] = sent[c].lower()
        sent[c] = re.sub(r'\W',' ',sent[c])
        sent[c] = re.sub(r'\s+',' ',sent[c])
    Cleaned_body.append(sent)
print(Cleaned_body[0])


# Creating the histogram

word2count = {}
for i in range(len(Cleaned_body)):
    for data in Cleaned_body[i]:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1


freq_word = heapq.nlargest(5000,word2count,key=word2count.get)

X = []
for i in range(len(Cleaned_body)):
    vector = []
    for data in Cleaned_body[i]:
        for word in freq_word:
            if word not in nltk.word_tokenize(data):
                vector.append(1)
            else:
                vector.append(0)
    X.append(vector)

X = np.asarray(X)
print(X)