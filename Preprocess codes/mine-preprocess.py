# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:07:36 2018

@author: hp
"""

import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from collections import Counter
from New_Uti import *
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


test=[]
train=[]
train_Text='train.txt'
test_Text='test.txt'
#train_Text='./FinalTrainFile.txt'
#test_Text='./FinalValidFile.txt'
with open(train_Text) as f:
    for line in f:
    	train.append(line)

f.close()


with open(test_Text) as f:
    for line in f:
    	test.append(line)

f.close()

train = [s.strip() for s in train]
test = [s.strip() for s in test]

text = train + test
text = [s.split(" ") for s in text]
train = [s.split(" ") for s in train]
test = [s.split(" ") for s in test]

def pad_sentences(sentences, sent, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    print(sequence_length)
    padded_sentences = []
    for i in range(len(sent)):
        sentence = sent[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

sentences_padded = pad_sentences(text, text)
sentences_padded_train = pad_sentences(text, train)
sentences_padded_test = pad_sentences(text, test)

sequence_length = max(len(x) for x in text)


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    #print(word_counts)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    #print(vocabulary_inv)
    vocabulary_inv = list(sorted(vocabulary_inv))
    #print(vocabulary_inv)
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #print(vocabulary)
    return [vocabulary, vocabulary_inv]

vocabulary, vocabulary_inv = build_vocab(sentences_padded)

train = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded_train])
test = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded_test])

train_path='train.csv'
dataframe=pd.read_csv(train_path)

X_train=dataframe['TEXT'].fillna('<UNK>').values
Y_train=dataframe['LABEL'].fillna('<UNK>').values

test_path='test.csv'
dataframe=pd.read_csv(test_path)

X_test=dataframe['TEXT'].fillna('<UNK>').values
Y_test=dataframe['LABEL'].fillna('<UNK>').values

X_train = [s.strip() for s in X_train]
X_test = [s.strip() for s in X_test]

x_text = X_train + X_test
#x_text = [s.split(" ") for s in x_text]

tokenizer = load_create_tokenizer(X_train,tok_path=None,savetokenizer=False)

embedding_dim = 100
l=train.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim1 = 5
word_index=tokenizer.word_index
emb_path = './glove.6B.100d.txt'
###file = open('glove.6B.100d.txt', encoding="utf8")
embedding_matrix = load_create_embedding_matrix(word_index,vocabulary_size,embedding_dim,emb_path,emb_pickle_path=False,save=False,saveName=None)


X_train=load_create_padded_data(X_train=X_train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path=None)
X_test=load_create_padded_data(X_train=X_test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path=None)

lbl_dict={}
index=0
for dial_lbls in Y_train:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1

def create_label(label):
	
    Y=[]
    for i in label:
    	xxx=np.zeros(int(len(lbl_dict)))
    	j=lbl_dict.get(i)
    	xxx[j]=1
    	Y.append(xxx)
    return Y

label = Y_train
Y_train = create_label(label)
label = Y_test
Y_test = create_label(label)

y_train=np.array([i for i in Y_train])
y_test=np.array([i for i in Y_test])

##dc = doc2Mat(X_train,emb_path,embedding_dim,vocabulary_size,None)