import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from collections import Counter
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

train_path='./train.csv'
dataframe=pd.read_csv(train_path)

X_train=dataframe['TEXT'].fillna('<UNK>').values
Y_train=dataframe['LABEL'].fillna('<UNK>').values

test_path='./test.csv'
dataframe=pd.read_csv(test_path)

X_test=dataframe['TEXT'].fillna('<UNK>').values
Y_test=dataframe['LABEL'].fillna('<UNK>').values

X_train = [s.strip() for s in X_train]
X_test = [s.strip() for s in X_test]

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

x_text = X_train + X_test
x_text = [clean_str(sent) for sent in x_text]
x_text = [s.split(" ") for s in x_text]


X_train = [clean_str(sent) for sent in X_train]
X_train = [s.split(" ") for s in X_train]
X_test = [clean_str(sent) for sent in X_test]
X_test = [s.split(" ") for s in X_test] 


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



###Generate labels

#y = np.concatenate([Y_train, Y_test], 0)


def create_label(label):
	
    Y=[]
    for i in range(0,len(label)):
    	if(label[i]=="CAT"):
    		Y.append([0,0,0,0,0,1])
    	if(label[i]=="QUESTION"):
    		Y.append([0,0,0,0,1,0])
    	if(label[i]=="INFO"):
    		Y.append([0,0,0,1,0,0])
    	if(label[i]=="COMMAND"):
    		Y.append([0,0,1,0,0,0])
    	if(label[i]=="G_G"):
    		Y.append([0,1,0,0,0,0])
    	if(label[i]!="G_G" and label[i]!="CAT"and label[i]!="QUESTION"and label[i]!="INFO"and label[i]!="COMMAND"):
    		Y.append([1,0,0,0,0,0])
    return Y

label = Y_train
Y_train = create_label(label)
label = Y_test
Y_test = create_label(label)

y_train=np.array([i for i in Y_train])
y_test=np.array([i for i in Y_test])
train = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded_train])
test = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded_test])

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = X_train.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 300

###########   FOR CNN
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

epochs = 100
batch_size = 50

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=6, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN-TRAINS-KIM/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adadelta(lr=0.01, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

predicted=model.predict(X_test)
print("DONE")


##################### for conv1d multi size filter CNN

filter_sizes = [3,4,5]
num_filters = 100
drop = 0.1

epochs = 100
batch_size = 50

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
#reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(embedding)
conv_1 = Conv1D(num_filters, kernel_size=filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(embedding)
conv_2 = Conv1D(num_filters, kernel_size=filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(embedding)
maxpool_0 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), padding='valid')(conv_0)
maxpool_1 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[1] + 1), padding='valid')(conv_1)
maxpool_2 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), padding='valid')(conv_2)
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
#flatten = Flatten()(concatenated_tensor)
layer1 = Bidirectional(LSTM(100))(concatenated_tensor)
dropout = Dropout(drop)(layer1)
output = Dense(units=43, activation='softmax')(dropout)
# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN1D-BILSTM-SWBDOLD/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

predicted=model.predict(X_test)
print("DONE")


##################### for results


results=[]
for i in predicted:
	pos=i.argmax()
	#print(pos)
	if(pos==0):
		results.append("blank")
	if(pos==1):
		results.append("G_G")
	if(pos==2):
		results.append("COMMAND")
	if(pos==3):
		results.append("INFO")
	if(pos==4):
		results.append("QUESTION")
	if(pos==5):
		results.append("CAT")



print(results)

#y_pred=np.array([i for i in results])

Y_test=[]
for i in y_test:
	pos=i.argmax()
	#print(pos)
	if(pos==0):
		Y_test.append("blank")
	if(pos==1):
		Y_test.append("G_G")
	if(pos==2):
		Y_test.append("COMMAND")
	if(pos==3):
		Y_test.append("INFO")
	if(pos==4):
		Y_test.append("QUESTION")
	if(pos==5):
		Y_test.append("CAT")

print(Y_test)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

print(accuracy_score(Y_test, results))
print(classification_report(Y_test, results,digits=4))
print(confusion_matrix(Y_test, results))





####################################### for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#embedding_vecor_length = 256
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")



##################### for CNN - LSTM

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


####################### for GRU

from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
embedding_vecor_length = 256
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(Dropout(0.2))
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


##################### for Bi-LSTM sequence labeling

from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
embedding_vecor_length = 256
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, input_shape=(sequence_length, embedding_dim), return_sequences=True)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint], batch_size=64, validation_data=(X_test, y_test))


################# for multiple filter CNN-LSTM

def squeeze_middle2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0], shape[1] * shape[2], shape[3] ] )
    return x3d

def squeeze_middle2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = ( in_batch, None, in_filters )
    else :
        output_shape = ( in_batch, in_rows * in_cols, in_filters )
    return output_shape

yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( xx )


