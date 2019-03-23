#Utility Functions to load the dataset, in case if pickle version of the files are there it will return so....
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import *
from keras.preprocessing.sequence import pad_sequences
from Text_Preprocessing import *
import timeit


def saveNumpyArrayCSV(numpy_array,file_name,delimiter=','):
	np.savetxt(file_name,numpy_array,delimiter=delimiter)

#Writes object_to_dump in file at location 'path'
def dumpPickle(path,object_to_dump):
	f=open(path,'w')
	pickle.dump(object_to_dump,f)
	f.close()

#Reads an object stores at 'path'
def readPickle(path):
	f=open(path)
	ob_to_load=pickle.load(f)
	f.close()
	return ob_to_load


#Loads an embedding matrix stores at emb_pickle_path if doesn't exists a pickled file, then creates One which is initialized by normal distribution with mean and std dev of the embeddings used, if given the option to save it saves
#the newly created embedding matrix at saveName
def load_create_embedding_matrix(word_index,vocab_size,emb_dim,emb_path,emb_pickle_path=False,save=False,saveName=None):
	if not emb_pickle_path:
		embedding_dict={}
		f=open(emb_path,'r')
		for line in f:
			fields=line.split()
			word=fields[0]
			#print(word)
			w_e=np.asarray(fields[1:],dtype='float32')
			embedding_dict[word]=w_e
		f.close()
		allembs=np.stack(embedding_dict.values())
		#print(allembs)
		emb_mean,emb_std=allembs.mean(),allembs.std()
		print(emb_mean)
		print(emb_std)
		embedding_matrix=np.random.normal(emb_mean,emb_std,(vocab_size,emb_dim))
		print(embedding_matrix)
		for word,index in word_index.items():
			vector=embedding_dict.get(word)
			if vector is not None:
				embedding_matrix[index]=vector
		if save:
			dumpPickle(saveName,embedding_matrix)
	else:
		f=open(emb_pickle_path)
		embedding_matrix=pickle.load(f)
		f.close()
	return embedding_matrix

#creates a tokenizer from training data file in csv format, if there is one it loads and returns
def load_create_tokenizer(train_data,tok_path=None,savetokenizer=False):
	if tok_path == None:
		tokenizer=Tokenizer()
		tokenizer.fit_on_texts(train_data)
		len(tokenizer.word_index)
		if savetokenizer:
			dumpPickle('./New_Tokenizer.tkn',tokenizer)
	else:
		tokenizer=readPickle(tok_path)
	return tokenizer


#Loads a CSV file, prepares the training data by padding and converting the text data in format, which is suitable and compatible
#to train via Embedding Layers in Keras
def loadCompleteCSV(csv_path,x_labels,y_labels,na_val,maxlen,tok_path,savetokenizer):
	start=timeit.default_timer()
	dataframe=pd.read_csv(csv_path)
	X_train=dataframe[x_labels].fillna(na_val).values
	X_train=load_create_padded_data(X_train=X_train,savetokenizer=savetokenizer,maxlen=maxlen,tokenizer_path=tok_path)
	Y_train=dataframe[y_labels]
	stop=timeit.default_timer()
	print('Execution Time : ' + str(stop-start)+' Seconds, To read the CSV file')
	return X_train,Y_train.values

#Raw Train Data in textual format
def load_create_padded_data(X_train,savetokenizer,padPath=None,isPaddingDone=False,maxlen=None,tokenizer_path=None,save_new_padded_data=False,path_for_new_data=False):
	#print(tokenizer_path)
	if not isPaddingDone:
		print("in if")
		tokenizer=load_create_tokenizer(X_train,tok_path=tokenizer_path,savetokenizer=savetokenizer)
		#word_index=tokenizer.word_index
		X_train=tokenizer.texts_to_sequences(X_train)
		#print(X_train)
		X_train=pad_sequences(X_train,maxlen=maxlen)
		#print(X_train)
		if save_new_padded_data:
			print("in save if")
			dumpPickle(path_for_new_data,X_train)
	else:
		print("in else")
		X_train=readPickle(padPath)
	return X_train

#Evaluate Test Data after reading from test file which is in csv format, if already exists pass dataframe (yeah rough I know)
def evaluateTestData(model,testpath,datacolumns,fillna_vals,tokenizer_path,maxlen,columns,dataframe_labels,pred_filename,issaveexists=False,savepadded=False,paddedtestpath=None,isid=False,index=None):
	if not issaveexists:
		dataframe=pd.read_csv(testpath)
		X_test=dataframe[datacolumns].fillna(fillna_vals).values
		X_test=load_create_padded_data(X_train=X_test,savetokenizer=False,maxlen=maxlen,tokenizer_path=tokenizer_path)
	makePredictionFile(model=model,test_data=X_test,isid=isid,columns=columns,test_dataframe=dataframe,labels=dataframe_labels,filename=pred_filename,index=index)

#Compatible to kaggle format, look for more
def makePredictionFile(model,test_data,labels,filename,index,columns,isid=True,test_dataframe=None):
	probabilities=model.predict(test_data)
	submission_df = pd.DataFrame(columns=columns)
	if isid:
		submission_df['id'] = test_dataframe['id'].values 
	submission_df[labels] = probabilities 
	submission_df.to_csv(filename, index=index)

'''
doc : List of text
embedding_path : path containing address of saved matrix(emb_path) or if matrix has to be build from scratch, then pretrained embedidng file path(emb_path)
Convert Text Matrix into Embedding Matrix
TODO : Implement the load embedding matrix part
''' 
def doc2Mat(doc,emb_path,emb_dim,maxlen,tokenizer_path):
	#Get Doc in padded integer converted format
	#If not given tokenizer, it will create from scratch, else will get help from tokenizer
	X_text=load_create_padded_data(doc,savetokenizer=False,maxlen=maxlen,tokenizer_path=tokenizer_path,save_new_padded_data=False)
	print('Tokenizing Doc Done...')
	try:
		tokenizer=readPickle(tokenizer_path)
		word_index=tokenizer.word_index
		vacab_size=len(word_index)+1
	except:
		tokenizer=None
		word_index=None
	embedding_matrix=load_create_embedding_matrix(word_index,vacab_size,emb_dim,emb_path,emb_path)
	print('Done Embedding Matrix Processing...')
	matrix=[]
	total_len=len(X_text)
	i=1
	old=0
	for rows in X_text:
		printStatus(i,total_len)
		vector=np.asarray([])
		for indices in rows:
			vector=np.concatenate((vector,embedding_matrix[indices]))
		#Eventually Print The Status
		done=(i*100.0)/total_len
		if(not int(done)/10==old):
			print('Done : ' + str(done) + '%')
			old=old+1
		i=i+1
		matrix.append(vector)
	return np.asarray(matrix)
#Usage Example : matrix=doc2Mat(documents,'./Emb_Mat.mat',300,100,'./tokenizer.tkn')


#Grid Search for best C and gamma value for a SVM model in sklearn
from sklearn import svm, grid_search

def svc_paramter_selection(X,y, nfolds):
	cs=[0.001, .01, 0.1, 1, 10]
	gammas=[.001, .01, .1, 1]
	param_grid={'C':cs,'gamma':gammas}
	grid_search= GridSearchCV(svm.SVC(kernel='rbf'),param_grid,cv=nfolds)
	grid_search.fit(X,y)
	grid_search.best_params_
	return grid_search.best_params_