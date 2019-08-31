from __future__ import print_function
import ast
import os

def conllWrite(outputPath, sentences, headers):
    """
    Writes a sentences array/hashmap to a CoNLL format
    """
    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))
    fOut = open(outputPath, 'w')
    
    
    for sentence in sentences:
        fOut.write("#")
        fOut.write("\t".join(headers))
        fOut.write("\n")
        for tokenIdx in range(len(sentence[headers[0]])):
            aceData = [sentence[key][tokenIdx] for key in headers]
            fOut.write("\t".join(aceData))
            fOut.write("\n")
        fOut.write("\n")
        
        
def readCoNLL(inputPath, cols, commentSymbol=None, valTransformation=None):
    """
    Reads in a CoNLL file
    """
    sentences = []
    
    sentenceTemplate = {name: [] for name in cols.values()} #sentenceTemplate = {'tokens':[]   ,'POS':[]  }
    
    sentence = {name: [] for name in sentenceTemplate.keys()} #sentence = {'tokens': []  ,'POS': [] }
    
    newData = False
    
    for line in open(inputPath):
        line = line.strip()
        if len(line) == 0 or (commentSymbol != None and line.startswith(commentSymbol)):
            if newData:      
                sentences.append(sentence)
                    
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue
        
        splits = line.split(" ")
        for colIdx, colName in cols.items():    # cols = { 0 : 'tokens' , 1: 'POS' }
            if (colName == 'tokens'):
                zz = splits[colIdx]
                zz = ast.literal_eval(zz)
                sentence[colName].append(zz)
                #for i in zz:
                    #sentence[colName].append(i)
            else:
                val = splits[colIdx]
                sentence[colName].append(val)
            
            if valTransformation != None:
                val = valTransformation(colName, val, splits)
                
    #        sentence[colName].append(val)  
                        
        newData = True
        #if newData:
            #sentences.append(sentence)
        
    if newData:        
        sentences.append(sentence)
            
                    
                   
    return sentences  



           
        