import pandas as pd

import nltk

train = pd.read_csv('C:/Users/anmol/Desktop/ML_Project/Dataset/Google_Quest/train.csv')
print(train.head(10))

Quest_body = train['question_body'].values
print(Quest_body[0])

for i in range (Quest_body.shape[0]):
    words = nltk.word_tokenize(Quest_body[i])
    tagged_words = nltk.pos_tag(words)
    named_ent = nltk.ne_chunk(tagged_words)
    print(named_ent)

