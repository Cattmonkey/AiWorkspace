# encoding=utf-8
import jieba
from oData import Data_Mysql
import codecs

#while True:
    #message = raw_input("Enter your message: ")
    #if message == "quit":
        #exit()
    #else: 
        #analysisResult = jieba.cut(message, cut_all=True)
        #print("Full Mode: " + "/ ".join(analysisResult))  # 全模式
        
        #analysisResult = jieba.cut(message, cut_all=False)
        #print("Default Mode: " + "/ ".join(analysisResult))  # 精确模式
        
        ##analysisResult = jieba.cut(message)  # 默认是精确模式
        ##print("Default Mode: " + ", ".join(analysisResult))
        
        #analysisResult = jieba.cut_for_search(message)  # 搜索引擎模式
        #print("cut_for_search: " + ", ".join(analysisResult))        
        ##print analysisResult


#值得详细研究的模式是精确模式，以及其用于识别新词的HMM模型和Viterbi算法。

#jieba.cut(sentence, cut_all, HMM):sentence-需要分词的字符串；cut_all-控制是否采用全模式；HMM-控制是否使用HMM模型；jieba.cut()返回的结构是一个可迭代的 generator。
#jieba.cut_for_search(sentence, HMM):sentence-需要分词的字符串；HMM-控制是否使用HMM模型；这种分词方法粒度比较细，成为搜索引擎模式；jieba.cut_for_search()返回的结构是一个可迭代的 generator。
#jieba.lcut()以及jieba.lcut_for_search用法和上述一致，最终返回的结构是一个列表list。

class Jieba ():
    def __init__ (self):
        pass
    def getData (self, table, colum, limit=''):
        readMysql = Data_Mysql ()
        contents = readMysql.read (table, colum, limit)
        print "content : ", contents   
        return contents
    def doJieba (self, indata, titleList, outPutName):
        for title in titleList:
            for line in indata[title]:
                if not line:
                    continue
                seg = jieba.cut(line.replace("\r\n", "").replace("/n", "").replace("/r", "").strip(), cut_all = False)
                s= ' '.join(seg)
                m=list(s)
                with open(outPutName,'a+')as f:
                    for word in m:
                        f.write(word.encode('utf-8'))        

if __name__ == "__main__":
    
    myjieba = Jieba ()
    myjieba.doJieba(myjieba.getData ('AutohomeCar_New', "*"), ["carName", "cartype", 'price', 'minprice', 'originalprice','Color', 'carEngine', 'carBody','carScore','carDetail'], "autohomeNew_seg.txt")
    
    
    #readMysql = Data_Mysql ()
    #contents = readMysql.read ("AutohomeContent", "content", "LIMIT 10000")
    #print "content : ", contents
    #for line in contents['content']:
        #if not line:
            #continue
        #seg = jieba.cut(line.replace("\r\n", "").replace("/n", "").replace("/r", "").strip(), cut_all = False)
        #s= ' '.join(seg)
        #m=list(s)
        #with open('result_seg.txt','a+')as f:
            #for word in m:
                #f.write(word.encode('utf-8'))
                #print word    
    