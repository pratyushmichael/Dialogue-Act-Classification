import pandas as pd
import numpy as np
import os
import pickle
import re
import itertools
from collections import Counter
from New_Utils import *
from keras.layers import *
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import *
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

train_path='./SWBD_train.csv'
dataframe=pd.read_csv(train_path)
f=open('trainingfile.txt','r')
i=0
X_train=[]
for line in f:
	line=line.strip()
	max1=len(line)
	if max1==0:
		#print(i)
		X_train.append("<UNK>")
	else:
		X_train.append(line)
	i=i+1


#X_train=dataframe['TEXT'].fillna('<UNK>').values
Y_train=dataframe['LABEL_OLD'].fillna('<UNK>').values

test_path='./SWBD_test.csv'
dataframe=pd.read_csv(test_path)

f=open('testingfile.txt','r')
i=0
X_test=[]
for line in f:
	line=line.strip()
	max1=len(line)
	if max1==0:
		#print(i)
		X_test.append("<UNK>")
	else:
		X_test.append(line)
	i=i+1


#X_test=dataframe['TEXT'].fillna('<UNK>').values
Y_test=dataframe['LABEL_OLD'].fillna('<UNK>').values

X_train = [s.strip() for s in X_train]
X_test = [s.strip() for s in X_test]

x_text = X_train

x_text = [s.split(" ") for s in x_text]

sequence_length = 0
for x in x_text:
	max1=len(x)
	if(max1>sequence_length):
		sequence_length=max1


f=open('New_Tokenizer.tkn','r')
tokenizer=pickle.load(f)
f.close()
f=open('Emb_Mat_swbd_ggl.mat','r')
embedding_matrix=pickle.load(f)
f.close()

tokenizer=load_create_tokenizer(X_train,None,True)
word_index=tokenizer.word_index
X_train=load_create_padded_data(X_train=X_train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
X_test=load_create_padded_data(X_train=X_test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,300,'./glove.840B.300d.txt',False,True,'./Emb_Mat_new.mat')


lbl_dict={}
index=0
for dial_lbls in Y_train:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1


'''

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
'''

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


#sequence_length = 51 # 56
#vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 300
#l=train.shape[1]
#embedding_dim1 = 5

##### for multi-filter size CNN

filter_sizes = [3,4,5]
num_filters = 100
drop = 0.1

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
output = Dense(units=43, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN-SWBDOLD-KIM/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='max')
adam = Adam(lr=0.01, decay=0.3)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
predicted=model.predict(X_test)
print("DONE")

################### for results-------------common to all

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


########################### for LSTM


from keras.models import Sequential
from keras.layers import *
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#embedding_vecor_length = 256
def create_model(drop=0.0,learn_rate=1e-4,dr=0.0,units=50):
	model = Sequential()
	model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
	model.add(Dropout(drop))
	model.add(LSTM(units))
	model.add(Dropout(drop))
	model.add(Dense(6, activation='softmax'))
	#checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max', baseline=None)
	adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=dr)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())
	return model

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max', baseline=None)
model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=64, validation_data=(X_test, y_test),verbose=1)
drop=[0.0,0.1,0.3,0.5]
learn_rate=[1e-4,0.001, 0.01, 0.1]
dr=[0.0,0.3,0.5]
units=[50,100,150]
param_grid = dict(drop=drop, learn_rate=learn_rate, dr=dr, units=units)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#fit_params=dict(callbacks=callbacks_list))
grid_result = grid.fit(X_train, y_train, fit_params=dict(callbacks=[checkpoint,stop]))
#model.fit(X_train, y_train, epochs=1, callbacks=[checkpoint,stop], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


####################### for fixed size filter CNN-LSTM

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=1024, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./CNN-2LSTM-SWBD/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.7)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=100, callbacks=[checkpoint, stop], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")

#################### for Bi-LSTM

from keras.models import Sequential
from keras.layers import *
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#embedding_vecor_length = 256
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(200)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./BILSTM-SWBD/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = RMSprop(lr=0.01, decay=0.7)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint,stop], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


################ for fixed size filter CNN-Bi-LSTM


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(GRU(100)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./MCNN-BIGRU-SWBD/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=200, callbacks=[checkpoint], batch_size=50, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


#################### for GRU


from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
embedding_vecor_length = 256
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Dropout(0.2))
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max', baseline=None)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint,stop], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")



######################## for fixed size filter CNN-GRU


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(100))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./CNN-GRU-SWBD-GGL/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.1, decay=0.7)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=100, callbacks=[checkpoint,stop], batch_size=64, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")



################### for multi-size filter CNN-LSTM
from keras.layers import Layer, Input, Lambda
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


filter_sizes = [3,4,5]
num_filters = 512
drop = 0.1

epochs = 100
batch_size = 64

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
yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( concatenated_tensor )
dropout = Dropout(drop)(yy)
layer_1 = LSTM(100)(dropout)
output = Dense(units=6, activation='softmax')(layer_1)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN-LSTM-TRAINS/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint,stop], validation_data=(X_test, y_test))  # starts training
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
predicted=model.predict(X_test)
print("DONE")


################### for multi-size filter CNN-Bi-LSTM


from keras.layers import Layer, Input, Lambda
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

filter_sizes = [3,4,5]
num_filters = 100
drop = 0.1

epochs = 200
batch_size = 50

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
yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( concatenated_tensor )
dropout = Dropout(drop)(yy)
layer_1 = Bidirectional(GRU(100))(dropout)
output = Dense(units=43, activation='softmax')(layer_1)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN-BIGRU-SWBDOLD/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


################### for multi-size filter CNN-GRU
from keras.layers import Layer, Input, Lambda
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence



filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

epochs = 100
batch_size = 64

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

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( concatenated_tensor )
dropout = Dropout(drop)(yy)
layer_1 = Bidirectional(GRU(100))(dropout)
output = Dense(units=43, activation='softmax')(layer_1)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./MCNN-BIGRU-NEW-NO WORD-OLDLABEL/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adagrad(lr=0.01, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")



##########################################
f=open('trainingfile.txt','r')
i=0
x=[]
for line in f:
	line=line.strip()
	max1=len(line)
	if max1==0:
		print(i)
		x.append("<UNK>")
	else:
		x.append(line)
	i=i+1
	



#################################################### for MCNN-BIGRU sequential


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
model = Sequential()
model.add(Merge([model1,model2,model3],mode='concat'))
model.add(Reshape((1,embedding_dim)))
model.add(Bidirectional(GRU(100)))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))
checkpoint = ModelCheckpoint('./CNN-BIGRU-TRAINS-POS/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit([X_train,X_train,X_train], y_train, epochs=200, callbacks=[checkpoint], batch_size=50, validation_data=([X_test,X_test,X_test], y_test))
scores = model.evaluate([X_test,X_test,X_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict([X_test,X_test,X_test],test])
print("DONE")
