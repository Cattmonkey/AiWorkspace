# coding:utf-8
import sys
#reload(sys)
#sys.setdefaultencoding( "utf-8" )
from gensim.models import Word2Vec
import tensorflow
#import logging,gensim,os
 
 
if __name__ == "__main__":
    #模型的加载
    model = Word2Vec.load('autohomeNew.model')
    print(model['动力'])
    #比较两个词语的相似度,越高越好
    #print('"轮胎" 和 "动力" 的相似度:'+ str(model.similarity('轮胎','动力')))
    #print('"轮胎" 和 "加速" 的相似度:'+ str(model.similarity('轮胎','加速')))    
    #使用一些词语来限定,分为正向和负向的
    result = model.most_similar(positive=['大型'])
    print('同"大型"接近词有:')
    for item in result:
        print('   "'+item[0]+'"  相似度:'+str(item[1]))
        
    result = model.most_similar(positive=['1.5', '手动'], negative=['北极'])
    print('同"1.5 L"与"手动"二词接近,但是与"北极"不接近词有:')
    for item in result:
        print('   "'+item[0]+'"  相似度:'+str(item[1]))    
     
    #result = model.most_similar(positive=['男人','权利'], negative=['女人'])
    #print('同"男人"和"权利"接近,但是与"女人"不接近的词有:')
    #for item in result:
        #print('   "'+item[0]+'"  相似度:'+str(item[1]))
     
    #result = model.most_similar(positive=['女人','法律'], negative=['男人'])
    #print('同"女人"和"法律"接近,但是与"男人"不接近的词有:')
    #for item in result:
        #print('   "'+item[0]+'"  相似度:'+str(item[1]))
    ##从一堆词里面找到不匹配的
    #print("有哪个是不匹配的? word2vec结果说是:"+model.doesnt_match("五层 刹车 冷却液 发动机".split()))
    #print("汽车 火车 单车 相机 , 有哪个是不匹配的? word2vec结果说是:"+model.doesnt_match("汽车 火车 单车 相机".split()))
    #print("大米 白色 蓝色 绿色 红色 , 有哪个是不匹配的? word2vec结果说是:"+model.doesnt_match("大米 白色 蓝色 绿色 红色 ".split()))
    ##直接查看某个词的向量
    #print('中国的特征向量是:')
    #print(model['中国'])
    
    #必须标记词性！！！！