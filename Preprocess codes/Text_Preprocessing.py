#import HTMLParser
import sys
import re
import numpy as np
#from New_Utils import *

#html_parser = HTMLParser.HTMLParser()
#reload(sys)
#sys.setdefaultencoding('utf8')
#Dictionary for mapping contractions
APPOSTOPHES={
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
} ## Need a huge dictionary

#manual function to check if a number is there in a string or not ?
def isNumber(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except(TypeError,ValueError):
		pass
	return False

import unicodedata
from string import digits
import sys

def printStatus(current,total):
	done=(current*100.0)/total
	sys.stdout.write('\rDone : %.2f%%'%(done))
	sys.stdout.flush()


def cleanText(X,saveCleaned=False,debug=False,saveName=None):
	kachra_2=[]
	regex_url = re.compile(r'http[s]?:[//|\\](?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	regex_hexadecimal=re.compile(r"[\x80-\xff]")
	clean_text=[]
	totallen=len(X)
	i=1
	for text in X:
		printStatus(i,totallen)
		i=i+1
		text=text.lower()
		text=text.replace('"', "'")
		text=text.encode('ascii','ignore')
		#text = html_parser.unescape(text)
		#text = text.decode("utf8").encode('ascii','ignore')
		urls_list=regex_url.findall(text)
		for url in urls_list:
			text=text.replace(url,"")
		hexa_list=regex_hexadecimal.findall(text)
		for hexa in hexa_list:
			text=text.replace(hexa,"")
		text=re.sub("[^\w\d'\s]+",' ',text)
		#remove digits and numbers from text :( "VERIFY THIS ONE !!! SUSPICIOUS BEHAVIOR MAYBE"
		text = text.translate(None, digits)
		text=re.sub("''+",' ',text)
		text=re.sub("'[ ]+"," ",text)
		text=re.sub("[ ]+'"," ",text)
		#text=re.sub(r"[!@#$%^&*\(\)\+=\{\}\[\]\.,\?_('')~:-/;|\\<>\"=]"," ",text)
		text=re.sub('[ ]+',' ',text)
		words = text.split()
		#reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
		reformed=[]
		for word in words:
			word_list=[]
			if word in APPOSTOPHES:
				word_list=APPOSTOPHES[word].split(" ")
				for word in word_list:
					reformed.append(word)
			else:
				reformed.append(word)
		clean_words=[]
		for word in reformed:
			if word.isalpha()==False:
				kachra_2.append(word)
			else:
				clean_words.append(word)
		text = " ".join(clean_words)
		#kachra.append(re.findall(r"[\x90-\xff]",text))
		clean_text.append(text)
	if(saveCleaned):
		saveCleanedText(saveName,kachra_2,clean_text,debug)
	return clean_text

def saveCleanedText(saveName,kachra_2,clean_text,debug):
	ff=set(kachra_2)
	len(ff)
	if debug:
		f=open('kachra_2.txt','w')
		for x in ff:
			f.write(str(x)+'\n')
		f.close()
	clean_text=np.asarray(clean_text)
	df=pd.DataFrame.to_csv(pd.DataFrame(clean_text),saveName,index=None)




'''
import pandas as pd
from New_Utils import *
from Tetx_Preprocessing import *
df=pd.read_csv('./test.csv')
X_test=df['comment_text'].fillna('<UNK>').values
clean_test=cleanText(X_test,saveCleaned=True,debug=False,saveName='CleanedTest.csv')

'''