# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:46:35 2018

@author: hp
"""
import ast
cols = {0:'tokens',1:'POS'}
sentences = []
    
sentenceTemplate = {name: [] for name in cols.values()} #sentenceTemplate = {'tokens':[]   ,'POS':[]  }
    
sentence = {name: [] for name in sentenceTemplate.keys()} #sentence = {'tokens': []  ,'POS': [] }
    
newData = False
commentSymbol = None
valTransformation = None
    
inputPath ='data/pod_m1/train.txt'
    

    for line in open(inputPath):
        line = line.strip()
        if len(line) == 0 or (commentSymbol != None and line.startswith(commentSymbol)):
            if newData:      
                sentences.append(sentence)
                    
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue
        
        splits = line.split("#")
        
        
        for colIdx, colName in cols.items():    # cols = { 0 : 'tokens' , 1: 'POS' }
            
            if (colName == 'tokens'):
                zz = splits[colIdx]
                #zz = ast.literal_eval(zz)
                sentence[colName].append(zz)
                #val = []
                #for i in zz:
                    #sentence[colName].append(i)
            else:
                val = splits[colIdx]
                sentence[colName].append(val)
            
            if valTransformation != None:
                val = valTransformation(colName, val, splits)
                
            #sentence[colName].append(val)  
                        
        newData = True
        #if newData:
            #sentences.append(sentence)
        
    if newData:        
        sentences.append(sentence)