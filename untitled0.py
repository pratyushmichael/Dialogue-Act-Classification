# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:55:44 2018

@author: hp
"""

trainRanges = []
trainData = []
trainData = data['pod1']['trainMatrix']
                oldSentLength = len(trainData[0]['tokens'])   #smallest token length         
                idxStart = 0
                
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx)) # [(0,1),....]
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))
                
                 miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(32))) #miniBatchSize: 32
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))               
  
for z in range(5,6):
    print("kl")                     