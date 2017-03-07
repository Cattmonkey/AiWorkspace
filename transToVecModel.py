# coding:utf-8
import sys
#reload(sys)
#setdefaultencoding( "utf-8" )
from gensim.models import Word2Vec
import logging,gensim,os
 
class TextLoader(object):
    def __init__(self, outfile):
        self.outfile = outfile
        pass
 
    def __iter__(self):
        input = open(self.outfile,'r', encoding= 'utf-8')
        line = str(input.readline())
        counter = 0
        while line!=None and len(line) > 4:
            #print line
            segments = line.split(' ')
            yield  segments
            line = str(input.readline())
 
if __name__ == "__main__":
    sentences = TextLoader('result_seg.txt')
    model = gensim.models.Word2Vec(sentences, workers=8)
    model.save('autohomeContent.model')
    
    sentences = TextLoader('autohomeNew_seg.txt')
    model = gensim.models.Word2Vec(sentences, workers=8)
    model.save('autohomeNew.model')    
    print ('ok')