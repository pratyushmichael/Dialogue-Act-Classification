import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from collections import Counter
from New_Utils import *
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
train_Text='./OFtrainnew.txt'
test_Text='./OFtestnew.txt'
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

train_path='./TRAINS_train.csv'
dataframe=pd.read_csv(train_path)

X_train=dataframe['TEXT'].fillna('<UNK>').values
Y_train=dataframe['LABEL'].fillna('<UNK>').values

test_path='./TRAINS_test.csv'
dataframe=pd.read_csv(test_path)

X_test=dataframe['TEXT'].fillna('<UNK>').values
Y_test=dataframe['LABEL'].fillna('<UNK>').values

X_train = [s.strip() for s in X_train]
X_test = [s.strip() for s in X_test]

x_text = X_train + X_test
#x_text = [s.split(" ") for s in x_text]

f=open('New_Tokenizer.tkn','r')
tokenizer=pickle.load(f)
f.close()
f=open('Emb_Mat_new.mat','r')
embedding_matrix=pickle.load(f)
f.close()

word_index=tokenizer.word_index
X_train=load_create_padded_data(X_train=X_train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
X_test=load_create_padded_data(X_train=X_test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')

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

embedding_dim = 300
l=train.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim1 = 5



#####################3 for fixed filter CNN-LSTM POS/CLUSTER

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

model1 = Sequential()
model1.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model2 = Sequential()
model2.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim1, input_length=sequence_length))
model1.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model2.add(MaxPooling1D(pool_size=2))
model = Sequential()
model.add(Merge([model1,model2],mode='concat'))
model.add(Bidirectional(GRU(100)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./CNN-BIGRU-TRAINS-CLUSTER/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit([X_train,train], y_train, epochs=200, callbacks=[checkpoint], batch_size=50, validation_data=([X_test,test], y_test))
scores = model.evaluate([X_test,test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict([X_test,test],test])
print("DONE")


############################# for MCNN conv2d-BI-LSTM/GRU sequential POS


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

epochs = 100
batch_size = 50

model1 = Sequential()
model1.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model2 = Sequential()
model2.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model3 = Sequential()
model3.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model1.add(Reshape((sequence_length,embedding_dim,1)))
model2.add(Reshape((sequence_length,embedding_dim,1)))
model3.add(Reshape((sequence_length,embedding_dim,1)))
model1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu'))
model2.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu'))
model3.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu'))
model1.add(MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid'))
model2.add(MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid'))
model3.add(MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid'))
model4 = Sequential()
model4.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim1, input_length=sequence_length))
model4.add(Reshape((sequence_length,embedding_dim1,1)))
model4.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim1), padding='valid', kernel_initializer='normal', activation='relu'))
model4.add(MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid'))
model5 = Sequential()
model5.add(Merge([model1,model2,model3],mode='concat'))
model = Sequential()
model.add(Merge([model5,model4],mode='concat'))
model.add(Reshape((1,400)))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./MCNN2D-BILSTM-TRAINS-POS/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit([X_train,X_train,X_train, train], y_train, epochs=200, callbacks=[checkpoint], batch_size=50, validation_data=([X_test,X_test,X_test,test], y_test))
scores = model.evaluate([X_test,X_test,X_test,test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict([X_test,X_test,X_test],test])
print("DONE")

####################### for MCNN conv1d-BI-LSTM/GRU sequential POS



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

epochs = 100
batch_size = 50

model1 = Sequential()
model1.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model2 = Sequential()
model2.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model3 = Sequential()
model3.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model1.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu'))
model2.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu'))
model3.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu'))
model1.add(MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), padding='valid'))
model2.add(MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), padding='valid'))
model3.add(MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), padding='valid'))
model4 = Sequential()
model4.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim1, input_length=sequence_length))
model4.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu'))
model4.add(MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), padding='valid'))
model5 = Sequential()
model5.add(Merge([model1,model2,model3],mode='concat'))
model = Sequential()
model.add(Merge([model5,model4],mode='concat'))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./MCNN1D-BILSTM-TRAINS-POS/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit([X_train,X_train,X_train, train], y_train, epochs=200, callbacks=[checkpoint], batch_size=50, validation_data=([X_test,X_test,X_test,test], y_test))
scores = model.evaluate([X_test,X_test,X_test,test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict([X_test,X_test,X_test],test])
print("DONE")


