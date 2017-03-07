# AI #

NLP知识库

- 基于自然语言技术的人工智能知识库

模块服务

## 自然语言处理工具包spaCy介绍
spaCy 是一个Python自然语言处理工具包，诞生于2014年年中，号称“Industrial-Strength Natural Language Processing in Python”，是具有工业级强度的Python NLP工具包。spaCy里大量使用了 Cython 来提高相关模块的性能，这个区别于学术性质更浓的Python NLTK，因此具有了业界应用的实际价值。

## Python 的十个自然语言处理工具

1.NLTK

NLTK 在用 Python 处理自然语言的工具中处于领先的地位。它提供了 WordNet 这种方便处理词汇资源的借口，还有分类、分词、除茎、标注、语法分析、语义推理等类库。

网站

http://www.nltk.org/

安装

安装 NLTK:

sudo pip install -U nltk
安装 Numpy (可选):

sudo pip install -U numpy
安装测试:

python then type import nltk
2.Pattern

Pattern 的自然语言处理工具有词性标注工具(Part-Of-Speech Tagger)，N元搜索(n-gram search)，情感分析(sentiment analysis)，WordNet。支持机器学习的向量空间模型，聚类，向量机。

网站:

https://github.com/clips/pattern

安装:

pip install pattern
3.TextBlob

TextBlob 是一个处理文本数据的 Python 库。提供了一些简单的api解决一些自然语言处理的任务，例如词性标注、名词短语抽取、情感分析、分类、翻译等等。

网站：

http://textblob.readthedocs.org/en/dev/

安装：

pip install -U textblob
4.Gensim

Gensim 提供了对大型语料库的主题建模、文件索引、相似度检索的功能。它可以处理大于RAM内存的数据。作者说它是“实现无干预从纯文本语义建模的最强大、最高效、最无障碍的软件。”

网站：

https://github.com/piskvorky/gensim

安装：

pip install -U gensim
5.PyNLPI

它的全称是：Python自然语言处理库（Python Natural Language Processing Library，音发作: pineapple） 这是一个各种自然语言处理任务的集合，PyNLPI可以用来处理N元搜索，计算频率表和分布，建立语言模型。他还可以处理向优先队列这种更加复杂的数据结构，或者像 Beam 搜索这种更加复杂的算法。

安装：

LInux:

sudo apt-get install pymol
Fedora:

yum install pymol
6.spaCy

这是一个商业的开源软件。结合Python和Cython，它的自然语言处理能力达到了工业强度。是速度最快，领域内最先进的自然语言处理工具。

网站：

https://github.com/proycon/pynlpl

安装：

pip install spacy
7.Polyglot

Polyglot 支持对海量文本和多语言的处理。它支持对165种语言的分词，对196中语言的辨识，40种语言的专有名词识别，16种语言的词性标注，136种语言的情感分析，137种语言的嵌入，135种语言的形态分析，以及69中语言的翻译。

网站：

https://pypi.python.org/pypi/polyglot

安装

pip install polyglot
8.MontyLingua

MontyLingua 是一个自由的、训练有素的、端到端的英文处理工具。输入原始英文文本到 MontyLingua ，就会得到这段文本的语义解释。适合用来进行信息检索和提取，问题处理，回答问题等任务。从英文文本中，它能提取出主动宾元组，形容词、名词和动词短语，人名、地名、事件，日期和时间，等语义信息。

网站：

http://web.media.mit.edu/~hugo/montylingua/

9.BLLIP Parser

BLLIP Parser（也叫做Charniak-Johnson parser）是一个集成了产生成分分析和最大熵排序的统计自然语言工具。包括 命令行 和 python接口 。

10.Quepy

Quepy是一个Python框架，提供将自然语言转换成为数据库查询语言。可以轻松地实现不同类型的自然语言和数据库查询语言的转化。所以，通过Quepy，仅仅修改几行代码，就可以实现你自己的自然语言查询数据库系统。


## jieba

"结巴"中文分词：做最好的Python中文分词组件 "Jieba" 

Feature

### 支持三种分词模式：

- 精确模式，试图将句子最精确地切开，适合文本分析；

- 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；

- 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

- 支持繁体分词

- 支持自定义词典

在线演示

http://jiebademo.ap01.aws.af.cm/

(Powered by Appfog)

### Python 2.x 下的安装

全自动安装：easy_install jieba 或者 pip install jieba

半自动安装：先下载http://pypi.python.org/pypi/jieba/ ，解压后运行python setup.py install

手动安装：将jieba目录放置于当前目录或者site-packages目录

通过import jieba 来引用 （第一次import时需要构建Trie树，需要几秒时间）

### Python 3.x 下的安装

目前master分支是只支持Python2.x 的

Python3.x 版本的分支也已经基本可用： https://github.com/fxsjy/jieba/tree/jieba3k

git clone https://github.com/fxsjy/jieba.git
git checkout jieba3k
python setup.py install
Algorithm

基于Trie树结构实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG)

采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合

对于未登录词，采用了基于汉字成词能力的HMM模型，使用了Viterbi算法

功能 1)：分词

jieba.cut方法接受两个输入参数: 1) 第一个参数为需要分词的字符串 2）cut_all参数用来控制是否采用全模式

jieba.cut_for_search方法接受一个参数：需要分词的字符串,该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细

注意：待分词的字符串可以是gbk字符串、utf-8字符串或者unicode

jieba.cut以及jieba.cut_for_search返回的结构都是一个可迭代的generator，可以使用for循环来获得分词后得到的每一个词语(unicode)，也可以用list(jieba.cut(...))转化为list

### 代码示例( 分词 )

	#encoding=utf-8
	import jieba
	
	seg_list = jieba.cut("我来到北京清华大学",cut_all=True)
	print "Full Mode:", "/ ".join(seg_list) #全模式
	
	seg_list = jieba.cut("我来到北京清华大学",cut_all=False)
	print "Default Mode:", "/ ".join(seg_list) #精确模式
	
	seg_list = jieba.cut("他来到了网易杭研大厦") #默认是精确模式
	print ", ".join(seg_list)
	
	seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") #搜索引擎模式
	print ", ".join(seg_list)
	Output:
	
	【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
	
	【精确模式】: 我/ 来到/ 北京/ 清华大学
	
	【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
	
	【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
### 功能 2) ：添加自定义词典

开发者可以指定自己自定义的词典，以便包含jieba词库里没有的词。虽然jieba有新词识别能力，但是自行添加新词可以保证更高的正确率

用法： jieba.load_userdict(file_name) # file_name为自定义词典的路径

词典格式和dict.txt一样，一个词占一行；每一行分三部分，一部分为词语，另一部分为词频，最后为词性（可省略），用空格隔开

范例：

	之前： 李小福 / 是 / 创新 / 办 / 主任 / 也 / 是 / 云 / 计算 / 方面 / 的 / 专家 /
	
	加载自定义词库后：　李小福 / 是 / 创新办 / 主任 / 也 / 是 / 云计算 / 方面 / 的 / 专家 /
	
	自定义词典：https://github.com/fxsjy/jieba/blob/master/test/userdict.txt
	
	用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_userdict.py
	
	"通过用户自定义词典来增强歧义纠错能力" --- https://github.com/fxsjy/jieba/issues/14

### 功能 3) ：关键词提取

jieba.analyse.extract_tags(sentence,topK) #需要先import jieba.analyse

setence为待提取的文本

topK为返回几个TF/IDF权重最大的关键词，默认值为20

代码示例 （关键词提取）
	
	https://github.com/fxsjy/jieba/blob/master/test/extract_tags.py
###功能 4) : 词性标注

标注句子分词后每个词的词性，采用和ictclas兼容的标记法

用法示例

	>>> import jieba.posseg as pseg
	>>> words =pseg.cut("我爱北京天安门")
	>>> for w in words:
	...    print w.word,w.flag
	...
	我 r
	爱 v
	北京 ns
	天安门 ns
	功能 5) : 并行分词

原理：将目标文本按行分隔后，把各行文本分配到多个python进程并行分词，然后归并结果，从而获得分词速度的可观提升

基于python自带的multiprocessing模块，目前暂不支持windows

用法：

jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数

jieba.disable_parallel() # 关闭并行分词模式

例子： https://github.com/fxsjy/jieba/blob/master/test/parallel/test_file.py

实验结果：在4核3.4GHz Linux机器上，对金庸全集进行精确分词，获得了1MB/s的速度，是单进程版的3.3倍。

### 功能 6) : Tokenize：返回词语在原文的起始位置

注意，输入参数只接受unicode

	默认模式
	
	result = jieba.tokenize(u'永和服装饰品有限公司')
	for tk in result:
	    print "word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2])
	word 永和                start: 0                end:2
	word 服装                start: 2                end:4
	word 饰品                start: 4                end:6
	word 有限公司            start: 6                end:10
	搜索模式
	
	result = jieba.tokenize(u'永和服装饰品有限公司',mode='search')
	for tk in result:
	    print "word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2])
	word 永和                start: 0                end:2
	word 服装                start: 2                end:4
	word 饰品                start: 4                end:6
	word 有限                start: 6                end:8
	word 公司                start: 8                end:10
	word 有限公司            start: 6                end:10
### 功能 7) : ChineseAnalyzer for Whoosh搜索引擎

引用： from jieba.analyse import ChineseAnalyzer

用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_whoosh.py



### 其他词典

占用内存较小的词典文件 https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.small

支持繁体分词更好的词典文件 https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big

下载你所需要的词典，然后覆盖jieba/dict.txt 即可或者用jieba.set_dictionary('data/dict.txt.big')

模块初始化机制的改变:lazy load （从0.28版本开始）

jieba采用延迟加载，"import jieba"不会立即触发词典的加载，一旦有必要才开始加载词典构建trie。如果你想手工初始jieba，也可以手动初始化。

import jieba
jieba.initialize() #手动初始化（可选）
在0.28之前的版本是不能指定主词典的路径的，有了延迟加载机制后，你可以改变主词典的路径:

jieba.set_dictionary('data/dict.txt.big')
例子： https://github.com/fxsjy/jieba/blob/master/test/test_change_dictpath.py

### 分词速度

1.5 MB / Second in Full Mode

400 KB / Second in Default Mode

Test Env: Intel(R) Core(TM) i7-2600 CPU @ 3.4GHz；《围城》.txt

### 常见问题

1）模型的数据是如何生成的？https://github.com/fxsjy/jieba/issues/7

2）这个库的授权是? https://github.com/fxsjy/jieba/issues/2

更多问题请点击：https://github.com/fxsjy/jieba/issues?sort=updated&state=closed

## SnowNLP简介
SnowNLP是一个python写的类库，可以方便的处理中文文本内容，是受到了TextBlob的启发而写的，由于现在大部分的自然语言处理库基本都是针对英文的，于是写了一个方便处理中文的类库，并且和TextBlob不同的是，这里没有用NLTK，所有的算法都是自己实现的，并且自带了一些训练好的字典。注意本程序都是处理的unicode编码，所以使用时请自行decode成unicode。

### PyYAML 详细介绍
PyYAML是一个Python的YAML解析器。

YAML ="YAML Ain't Markup Language"（缩写为YAML）。这是一种数据序列化（serialization ）语言，是一种可读的文本的数据结构，它的设计目标是使人们容易读，程序容易处理。它类似XML，但是比XML简单。


### python NLTK 环境搭建

1.安装Python（我安装的是Python2.7.8，目录D:\Python27）

2.安装NumPy（可选）

到这里下载： http://sourceforge.net/projects/numpy/files/NumPy/1.6.2/numpy-1.6.2-win32-superpack-python2.7.exe

注意Py版本

下载之后执行exe文件（程序会自动搜索python27目录）

3.安装NLTK（我下载的是nltk-2.0.3）

到这里下载： http://pypi.python.org/pypi/nltk

把nltk-3.0.0解压到D:\Python27目录

打开cmd，进到D:\Python27\nltk-3.0.0目录（输入：cd D:\Python27\nltk-3.0.0）

输入命令：python setup.py install

这时出现 Import error :no module named setuptools windows 默认没有安装setuptool模块，自己下载这个模块.exe ( http://www.cr173.com/soft/40214.html#address )

4.安装PyYAML：

到这里下载： http://pyyaml.org/wiki/PyYAML

注意Py版本

下载之后执行exe文件（程序会自动搜索python27目录）

5.打开IDLE，输入import nltk，没有错误的话，就说明安装成功了。

到这里，NLP所需的基本python模块都已经安装好了，然后要安装NLTK_DATA了

下载NLTK_DATA有好几种方法，这里我只介绍一种

6.继续第五步，已经import nltk了，然后输入nltk.download()，这样就可以打开一个NLTK Downloader（NLTK下载器）

7.注意下载器下边的Download Directory，我设置的是C:\nltk_data

8.在计算机-属性-高级系统设置-高级-环境变量-系统变量-新建：上边：NLTK_DATA，下边：C:\nltk_data

9.选择你要下载的包（语料库、模块），可以一次性下载（我在下载过程中总是出现out of date），也可以逐个下载（这样速度比较快，整体下载速度很慢）

10.成功安装包之后怎么测试呢？输入下边的语句就可以。

	>>> from nltk.corpus import brown
	>>> brown.words()
	['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
还有一个python经典画图库matplotlib的安装： http://blog.csdn.net/huruzun/article/details/39395343

这些环境搭建好，基本常用的python需要的工具包之类的就已经完成啦！

## jieba分词学习笔记（一）

### 序
中科院的ICTCLAS，哈工大的ltp，东北大学的NIU Parser是学术界著名的分词器，我曾浅显读过一些ICTCLAS的代码，然而并不那么好读。jieba分词是python写成的一个算是工业界的分词开源库，其github地址为：https://github.com/fxsjy/jieba

jieba分词虽然效果上不如ICTCLAS和ltp，但是胜在python编写，代码清晰，扩展性好，对jieba有改进的想法可以很容易的自己写代码进行魔改。毕竟这样说好像自己就有能力改进jieba分词一样_(:з」∠)_

网上诸多关于jieba分词的分析，多已过时，曾经分析jieba分词采用trie树的数据结构云云的文章都已经过时，现在的jieba分词已经放弃trie树，采用前缀数组字典的方式存储词典。
本文分析的jieba分词基于2015年7月左右的代码进行，日后jieba若更新，看缘分更新这一系列文章_(:з」∠)_

###jieba分词的基本思路
jieba分词对已收录词和未收录词都有相应的算法进行处理，其处理的思路很简单，当然，过于简单的算法也是制约其召回率的原因之一。

其主要的处理思路如下：

- 加载词典dict.txt
- 从内存的词典中构建该句子的DAG（有向无环图）
- 对于词典中未收录词，使用HMM模型的viterbi算法尝试分词处理
- 已收录词和未收录词全部分词完毕后，使用dp寻找DAG的最大概率路径
- 输出分词结果
### 词典的加载
语料库和词典
jieba分词默认的模型使用了一些语料来做训练集，在 https://github.com/fxsjy/jieba/issues/7 中，作者说

来源主要有两个，一个是网上能下载到的1998人民日报的切分语料还有一个msr的切分语料。另一个是我自己收集的一些txt小说，用ictclas把他们切分（可能有一定误差）。 然后用python脚本统计词频。
jieba分词的默认语料库选择看起来满随意的_(:з」∠)_，作者也吐槽高质量的语料库不好找，所以如果需要在生产环境使用jieba分词，尽量自己寻找一些高质量的语料库来做训练集。

语料库中所有的词语被用来做两件事情：

对词语的频率进行统计，作为登录词使用
对单字在词语中的出现位置进行统计，使用BMES模型进行统计，供后面套HMM模型Viterbi算法使用，这个后面说。
统计后的结果保存在dict.txt中，摘录其部分结构如下：

	上访 212 v
	上访事件 3 n
	上访信 3 nt
	上访户 3 n
	上访者 5 n
	上证 120 j
	上证所 8 nt
	上证指数 3 n
	上证综指 3 n
	上诉 187 v
	上诉书 3 n
	上诉人 3 n
	上诉期 3 b
	上诉状 4 n
	上课 650 v
其中，第一列是中文词语，第二列是词频，第三列是词性，jieba分词现在的版本除了分词也提供词性标注等其他功能，这个不在本文讨论范围内，可以忽略第三列。jieba分词所有的统计来源，就是这个语料库产生的两个模型文件。

### 对字典的处理
jieba分词为了快速地索引词典以加快分词性能，使用了前缀数组的方式构造了一个dict用于存储词典。

在旧版本的jieba分词中，jieba采用trie树的数据结构来存储，其实对于python来说，使用trie树显得非常多余，我将对新老版本的字典加载分别进行分析。

### trie树
trie树简介

trie树又叫字典树，是一种常见的数据结构，用于在一个字符串列表中进行快速的字符串匹配。其核心思想是将拥有公共前缀的单词归一到一棵树下以减少查询的时间复杂度，其主要缺点是占用内存太大了。

trie树按如下方法构造：

trie树的根节点是空，不代表任何含义
其他每个节点只有一个字符，词典中所有词的第一个字的集合作为第一层叶子节点，以字符α开头的单词挂在以α为根节点的子树下，所有以α开头的单词的第二个字的集合作为α子树下的第一层叶子节点，以此类推
从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串
一个以and at as cn com构造的trie树如下图：



查找过程如下：

- 从根结点开始一次搜索；
- 取得要查找关键词的第一个字母，并根据该字母选择对应的子树并转到该子树继续进行检索；
- 在相应的子树上，取得要查找关键词的第二个字母,并进一步选择对应的子树进行检索。
- 迭代过程……
- 在某个结点处，关键词的所有字母已被取出，则读取附在该结点上的信息，即完成查找。其他操作类似处理.
- 如查询at，可以找到路径root-a-t的路径，对于单词av，从root找到a后，在a的叶子节点下面不能找到v结点，则查找失败。

trie树的查找时间复杂度为O(k)，k = len(s)，s为目标串。

二叉查找树的查找时间复杂度为O(lgn)，比起二叉查找树，trie树的查找和结点数量无关，因此更加适合词汇量大的情况。

但是trie树对空间的消耗是很大的，是一个典型的空间换时间的数据结构。

jieba分词的trie树

旧版本jieba分词中关于trie树的生成代码如下：

	def gen_trie(f_name):  
	    lfreq = {}  
	    trie = {}  
	    ltotal = 0.0  
	    with open(f_name, 'rb') as f:  
	        lineno = 0   
	        for line in f.read().rstrip().decode('utf-8').split('\n'):  
	            lineno += 1  
	            try:  
	                word,freq,_ = line.split(' ')  
	                freq = float(freq)  
	                lfreq[word] = freq  
	                ltotal+=freq  
	                p = trie  
	                for c in word:  
	                    if c not in p:  
	                        p[c] ={}  
	                    p = p[c]  
	                p['']='' #ending flag  
	            except ValueError, e:  
	                logger.debug('%s at line %s %s' % (f_name,  lineno, line))  
	                raise ValueError, e  
	    return trie, lfreq, ltotal  
代码很简单，遍历每行文件，对于每个单词的每个字母，在trie树（trie和p变量）中查找是否存在，如果存在，则挂到下面，如果不存在，就建立新子树。

jieba分词采用python 的dict来存储树，这也是python对树的数据结构的通用做法。

我写了一个函数来直观输出其生成的trie树，代码如下：


	def print_trie(tree, buff, level = 0, prefix=''):
	    count = len(tree.items())
	    for k,v in tree.items():
	        count -= 1
	        buff.append('%s +- %s' % ( prefix , k if k!='' else 'NULL'))
	        if v:
	            if count  == 0:
	                print_trie(v, buff, level + 1, prefix + '    ')
	            else:
	                print_trie(v, buff, level + 1, prefix + ' |  ')
	        pass
	    pass
	
	trie, list_freq, total =  gen_trie('a.txt')
	buff = ['ROOT']
	print_trie(trie, buff, 0)
	print('\n'.join(buff))
使用上面列举出的dict.txt的部分词典作为样例，输出结果如下

	ROOT
	 +- 上
	     +- 证
	     |   +- NULL
	     |   +- 所
	     |   |   +- NULL
	     |   +- 综
	     |   |   +- 指
	     |   |       +- NULL
	     |   +- 指
	     |       +- 数
	     |           +- NULL
	     +- 诉
	     |   +- NULL
	     |   +- 人
	     |   |   +- NULL
	     |   +- 状
	     |   |   +- NULL
	     |   +- 期
	     |   |   +- NULL
	     |   +- 书
	     |       +- NULL
	     +- 访
	     |   +- NULL
	     |   +- 信
	     |   |   +- NULL
	     |   +- 事
	     |   |   +- 件
	     |   |       +- NULL
	     |   +- 者
	     |   |   +- NULL
	     |   +- 户
	     |       +- NULL
	     +- 课
	         +- NULL
### 使用trie树的问题

本来jieba采用trie树的出发点是可以的，利用空间换取时间，加快分词的查找速度，加速全切分操作。但是问题在于python的dict原生使用哈希表实现，在dict中获取单词是近乎O(1)的时间复杂度，所以使用trie树，其实是一种避重就轻的做法。

于是2014年某位同学的PR修正了这一情况。

前缀数组
在2014年的某次PR中（https://github.com/fxsjy/jieba/pull/187 ），提交者将trie树改成前缀数组，大大地减少了内存的使用，加快了查找的速度。

现在jieba分词对于词典的操作，改为了一层word:freq的结构，存于lfreq中，其具体操作如下：

对于每个收录词，如果其在lfreq中，则词频累积，如果不在则加入lfreq
对于该收录词的所有前缀进行上一步操作，如单词'cat'，则对c, ca, cat分别进行第一步操作。除了单词本身的所有前缀词频初始为0.

	def gen_pfdict(self, f):
	        lfreq = {}
	        ltotal = 0
	        f_name = resolve_filename(f)
	        for lineno, line in enumerate(f, 1):
	            try:
	                line = line.strip().decode('utf-8')
	                word, freq = line.split(' ')[:2]
	                freq = int(freq)
	                lfreq[word] = freq
	                ltotal += freq
	                for ch in xrange(len(word)):
	                    wfrag = word[:ch + 1]
	                    if wfrag not in lfreq:
	                        lfreq[wfrag] = 0
	            except ValueError:
	                raise ValueError(
	                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
	        f.close()
	        return lfreq, ltotal

很朴素的做法，然而充分利用了python的dict类型，效率提高了不少。

## jieba分词学习笔记（一）
###分词模式
jieba分词有多种模式可供选择。可选的模式包括：

- 全切分模式
- 精确模式
- 搜索引擎模式
- 同时也提供了HMM模型的开关。

其中全切分模式就是输出一个字串的所有分词，

精确模式是对句子的一个概率最佳分词，

而搜索引擎模式提供了精确模式的再分词，将长词再次拆分为短词。

效果大抵如下：

	# encoding=utf-8
	import jieba
	
	seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
	print("Full Mode: " + "/ ".join(seg_list))  # 全模式
	
	seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
	print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
	
	seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
	print(", ".join(seg_list))
	
	seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
	print(", ".join(seg_list))
	的结果为
	
	【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
	
	【精确模式】: 我/ 来到/ 北京/ 清华大学
	
	【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
	
	【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
其中，新词识别即用HMM模型的Viterbi算法进行识别新词的结果。

值得详细研究的模式是精确模式，以及其用于识别新词的HMM模型和Viterbi算法。

jieba.cut()
在载入词典之后，jieba分词要进行分词操作，在代码中就是核心函数jieba.cut()，代码如下：

	 def cut(self, sentence, cut_all=False, HMM=True):
	        '''
	        The main function that segments an entire sentence that contains
	        Chinese characters into seperated words.
	        Parameter:
	            - sentence: The str(unicode) to be segmented.
	            - cut_all: Model type. True for full pattern, False for accurate pattern.
	            - HMM: Whether to use the Hidden Markov Model.
	        '''
	        sentence = strdecode(sentence)
	
	        if cut_all:
	            re_han = re_han_cut_all
	            re_skip = re_skip_cut_all
	        else:
	            re_han = re_han_default
	            re_skip = re_skip_default
	        if cut_all:
	            cut_block = self.__cut_all
	        elif HMM:
	            cut_block = self.__cut_DAG
	        else:
	            cut_block = self.__cut_DAG_NO_HMM
	        blocks = re_han.split(sentence)
	        for blk in blocks:
	            if not blk:
	                continue
	            if re_han.match(blk):
	                for word in cut_block(blk):
	                    yield word
	            else:
	                tmp = re_skip.split(blk)
	                for x in tmp:
	                    if re_skip.match(x):
	                        yield x
	                    elif not cut_all:
	                        for xx in x:
	                            yield xx
	                    else:
	                        yield x
其中，

docstr中给出了默认的模式，精确分词 + HMM模型开启。

第12-23行进行了变量配置。

第24行做的事情是对句子进行中文的切分，把句子切分成一些只包含能处理的字符的块（block），丢弃掉特殊字符，因为一些词典中不包含的字符可能对分词产生影响。

24行中re_han默认值为re_han_default，是一个正则表达式，定义如下：

	# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
	re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)
可以看到诸如空格、制表符、换行符之类的特殊字符在这个正则表达式被过滤掉。

25-40行使用yield实现了返回结果是一个迭代器，即文档中所说：

jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)
其中，31-40行，如果遇到block是非常规字符，就正则验证一下直接输出这个块作为这个块的分词结果。如标点符号等等，在分词结果中都是单独一个词的形式出现的，就是这十行代码进行的。

关键在28-30行，如果是可分词的block，那么就调用函数cut_block，默认是cut_block = self.__cut_DAG，进行分词

jieba.__cut_DAG()
__cut_DAG的作用是按照DAG，即有向无环图进行切分单词。其代码如下：

	def __cut_DAG(self, sentence):
	        DAG = self.get_DAG(sentence)
	        route = {}
	        self.calc(sentence, DAG, route)
	        x = 0
	        buf = ''
	        N = len(sentence)
	        while x < N:
	            y = route[x][1] + 1
	            l_word = sentence[x:y]
	            if y - x == 1:
	                buf += l_word
	            else:
	                if buf:
	                    if len(buf) == 1:
	                        yield buf
	                        buf = ''
	                    else:
	                        if not self.FREQ.get(buf):
	                            recognized = finalseg.cut(buf)
	                            for t in recognized:
	                                yield t
	                        else:
	                            for elem in buf:
	                                yield elem
	                        buf = ''
	                yield l_word
	            x = y
	
	        if buf:
	            if len(buf) == 1:
	                yield buf
	            elif not self.FREQ.get(buf):
	                recognized = finalseg.cut(buf)
	                for t in recognized:
	                    yield t
	            else:
	                for elem in buf:
	                    yield elem
对于一个sentence，首先 获取到其有向无环图DAG，然后利用dp对该有向无环图进行最大概率路径的计算。计算出最大概率路径后迭代，如果是登录词，则输出，如果是单字，将其中连在一起的单字找出来，这些可能是未登录词，使用HMM模型进行分词，分词结束之后输出。

至此，分词结束。

其中，值得跟进研究的是第2行获取DAG，第4行计算最大概率路径和第20和34行的使用HMM模型进行未登录词的分词，在后面的文章中会进行解读。
	
	DAG = self.get_DAG(sentence)
	
	    ...
	
	self.calc(sentence, DAG, route)
	
	    ...
	
	recognized = finalseg.cut(buf)
## jieba分词学习笔记（一）
### DAG（有向无环图）
有向无环图，directed acyclic graphs，简称DAG，是一种图的数据结构，其实很naive，就是没有环的有向图_(:з」∠)_

DAG在分词中的应用很广，无论是最大概率路径，还是后面套NN的做法，DAG都广泛存在于分词中。

因为DAG本身也是有向图，所以用邻接矩阵来表示是可行的，但是jieba采用了python的dict，更方便地表示DAG，其表示方法为:

	{prior1:[next1,next2...,nextN]，prior2:[next1',next2'...nextN']...}
以句子 "国庆节我在研究结巴分词"为例，其生成的DAG的dict表示为：

	{0: [0, 1, 2], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6], 6: [6], 7: [7, 8], 8: [8], 9: [9, 10], 10: [10]}

其中，

	国[0] 庆[1] 节[2] 我[3] 在[4] 研[5] 究[6] 结[7] 巴[8] 分[9] 词[10]
get_DAG()函数代码如下：

	def get_DAG(self, sentence):
	        self.check_initialized()
	        DAG = {}
	        N = len(sentence)
	        for k in xrange(N):
	            tmplist = []
	            i = k
	            frag = sentence[k]
	            while i < N and frag in self.FREQ:
	                if self.FREQ[frag]:
	                    tmplist.append(i)
	                i += 1
	                frag = sentence[k:i + 1]
	            if not tmplist:
	                tmplist.append(k)
	            DAG[k] = tmplist
	        return DAG
frag即fragment，可以看到代码循环切片句子，FREQ即是词典的{word:frequency}的dict

因为在载入词典的时候已经将word和word的所有前缀加入了词典，所以一旦frag not in FREQ，即可以断定frag和以frag为前缀的词不在词典里，可以跳出循环。

由此得到了DAG，下一步就是使用dp动态规划对最大概率路径进行求解。

### 最大概率路径
值得注意的是，DAG的每个结点，都是带权的，对于在词典里面的词语，其权重为其词频，即FREQ[word]。我们要求得route = (w1, w2, w3 ,.., wn)，使得Σweight(wi)最大。

### 动态规划求解法
满足dp的条件有两个

### 重复子问题
### 最优子结构
我们来分析最大概率路径问题。

#### 重复子问题
对于结点Wi和其可能存在的多个后继Wj和Wk，有:

	任意通过Wi到达Wj的路径的权重为该路径通过Wi的路径权重加上Wj的权重{Ri->j} = {Ri + weight(j)} ；
	任意通过Wi到达Wk的路径的权重为该路径通过Wi的路径权重加上Wk的权重{Ri->k} = {Ri + weight(k)} ；
	即对于拥有公共前驱Wi的节点Wj和Wk，需要重复计算到达Wi的路径。

#### 最优子结构
对于整个句子的最优路径Rmax和一个末端节点Wx，对于其可能存在的多个前驱Wi，Wj，Wk...,设到达Wi，Wj，Wk的最大路径分别为Rmaxi，Rmaxj，Rmaxk，有：

	Rmax = max(Rmaxi,Rmaxj,Rmaxk...) + weight(Wx)

于是问题转化为

求Rmaxi, Rmaxj, Rmaxk...

组成了最优子结构，子结构里面的最优解是全局的最优解的一部分。

状态转移方程
由上一节，很容易写出其状态转移方程

Rmax = max{(Rmaxi,Rmaxj,Rmaxk...) + weight(Wx)}

代码
上面理解了，代码很简单，注意一点total的值在加载词典的时候求出来的，为词频之和，然后有一些诸如求对数的trick，代码是典型的dp求解代码。

	def calc(self, sentence, DAG, route):
	        N = len(sentence)
	        route[N] = (0, 0)
	        logtotal = log(self.total)
	        for idx in xrange(N - 1, -1, -1):
	            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
	                              logtotal + route[x + 1][0], x) for x in DAG[idx])

----------------------
### jieba
### 1.分词
#### 1.1主要分词函数
jieba.cut(sentence, cut_all, HMM):sentence-需要分词的字符串；cut_all-控制是否采用全模式；HMM-控制是否使用HMM模型；jieba.cut()返回的结构是一个可迭代的 generator。
jieba.cut_for_search(sentence, HMM):sentence-需要分词的字符串；HMM-控制是否使用HMM模型；这种分词方法粒度比较细，成为搜索引擎模式；jieba.cut_for_search()返回的结构是一个可迭代的 generator。
jieba.lcut()以及jieba.lcut_for_search用法和上述一致，最终返回的结构是一个列表list。
#### 1.2示例

	import jieba as jb
	
	seg_list = jb.cut("我来到北京清华大学", cut_all=True)
	print("全模式: " + "/ ".join(seg_list))  # 全模式
	
	seg_list = jb.cut("我来到北京清华大学", cut_all=False)
	print("精确模式: " + "/ ".join(seg_list))  # 精确模式
	
	seg_list = jb.cut("他来到了网易杭研大厦")  
	print("默认模式: " + "/ ".join(seg_list)) # 默认是精确模式
	seg_list = jb.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  

	print("搜索引擎模式: " + "/ ".join(seg_list)) # 搜索引擎模式

---------------------- 100287112801661

## 如何用 Python 中的 NLTK 对中文进行分析和处理？

用nltk对自己的日记进行分析。得到以下结果（节选） '\xb8\xb0', '\xe5\xbc\xba\xe8\xba', '\xe5\xbd\xbc…显示全部

最近正在用nltk 对中文网络商品评论进行褒贬情感分类，计算评论的信息熵（entropy）、互信息（point mutual information）和困惑值（perplexity）等（不过这些概念我其实也还理解不深...只是nltk 提供了相应方法）。

我感觉用nltk 处理中文是完全可用的。其重点在于中文分词和文本表达的形式。
中文和英文主要的不同之处是中文需要分词。因为nltk 的处理粒度一般是词，所以必须要先对文本进行分词然后再用nltk 来处理（不需要用nltk 来做分词，直接用分词包就可以了。严重推荐结巴分词，非常好用）。
中文分词之后，文本就是一个由每个词组成的长数组：[word1, word2, word3…… wordn]。之后就可以使用nltk 里面的各种方法来处理这个文本了。比如用FreqDist 统计文本词频，用bigrams 把文本变成双词组的形式：[(word1, word2), (word2, word3), (word3, word4)……(wordn-1, wordn)]。
再之后就可以用这些来计算文本词语的信息熵、互信息等。
再之后可以用这些来选择机器学习的特征，构建分类器，对文本进行分类（商品评论是由多个独立评论组成的多维数组，网上有很多情感分类的实现例子用的就是nltk 中的商品评论语料库，不过是英文的。但整个思想是可以一致的）。

另外还有一个困扰很多人的Python 中文编码问题。多次失败后我总结出一些经验。
Python 解决中文编码问题基本可以用以下逻辑：
utf8（输入） ——> unicode（处理） ——> （输出）utf8
Python 里面处理的字符都是都是unicode 编码，因此解决编码问题的方法是把输入的文本（无论是什么编码）解码为（decode）unicode编码，然后输出时再编码（encode）成所需编码。
由于处理的一般为txt 文档，所以最简单的方法，是把txt 文档另存为utf-8 编码，然后使用Python 处理的时候解码为unicode（sometexts.decode('utf8')），输出结果回txt 的时候再编码成utf8（直接用str() 函数就可以了）。

另外这篇文章也有很详细的讲到nltk 的中文应用，很值得参考：http://blog.csdn.net/huyoo/article/details/12188573

-----------------------------------

## TensorFlow
TensorFlow(腾三福[1]  )是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。
TensorFlow可被用于语音识别或图像识别等多项机器深度学习领域，对2011年开发的深度学习基础架构DistBelief进行了各方面的改进，它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow将完全开源，任何人都可以用。
### 目录
- 1.支持算法
- 2.开源意义
- 3.中文文档

支持算法编辑
TensorFlow 表达了高层次的机器学习计算，大幅简化了第一代系统，并且具备更好的灵活性和可延展性。TensorFlow一大亮点是支持异构设备分布式计算，它能够在各个平台上自动运行模型，从手机、单个CPU / GPU到成百上千GPU卡组成的分布式系统。[2] 

从目前的文档看，TensorFlow支持CNN、RNN和LSTM算法，这都是目前在Image，Speech和NLP最流行的深度神经网络模型。

### 开源意义编辑
这一次的Google开源深度学习系统TensorFlow在很多地方可以应用，如语音识别，自然语言理解，计算机视觉，广告等等。但是，基于以上论点，我们也不能过分夸大TensorFlow这种通用深度学习框架在一个工业界机器学习系统里的作用。在一个完整的工业界语音识别系统里， 除了深度学习算法外，还有很多工作是专业领域相关的算法，以及海量数据收集和工程系统架构的搭建。

不过总的来说，这次谷歌的开源很有意义，尤其是对于中国的很多创业公司来说，他们大都没有能力理解并开发一个与国际同步的深度学习系统，所以TensorFlow会大大降低深度学习在各个行业中的应用难度。[2] 
中文文档编辑

官方文档中文版[3]  通过协同翻译，现已上线，国内的爱好者可以通过GitHub协作的方式查看并完善此中文版文档。
参考资料

1.  谷歌发布tensorflow(腾三福) 1.0  ．中国网．2017-02-22[引用日期2017-03-1]
2.  揭秘TensorFlow：Google开源到底开的是什么？  ．新浪[引用日期2015-11-12]
3.  TensorFlow官方文档中文版协同翻译库  ．GitHub[引用日期2015-12-30]

			Tensorflow搞一个聊天机器人
			catalogue
			
			0. 前言
			1. 训练语料库
			2. 数据预处理
			3. 词汇转向量
			4. 训练
			5. 聊天机器人 - 验证效果
			 
			
			0. 前言
			
			不是搞机器学习算法专业的，3个月前开始补了一些神经网络，卷积，神经网络一大堆基础概念，尼玛，还真有点复杂，不过搞懂这些基本数学概念，再看tensorflow的api和python代码觉得跌跌撞撞竟然能看懂了，背后的意思也能明白一点点
			
			0x1: 模型分类
			
			1. 基于检索的模型 vs. 产生式模型
			
			基于检索的模型(Retrieval-Based Models)有一个预先定义的"回答集(repository)"，包含了许多回答(responses)，还有一些根据输入的问句和上下文(context)，以及用于挑选出合适的回答的启发式规则。这些启发式规则可能是简单的基于规则的表达式匹配，或是相对复杂的机器学习分类器的集成。基于检索的模型不会产生新的文字，它只能从预先定义的"回答集"中挑选出一个较为合适的回答。
			产生式模型(Generative Models)不依赖于预先定义的回答集，它会产生一个新的回答。经典的产生式模型是基于机器翻译技术的，只不过不是将一种语言翻译成另一种语言，而是将问句"翻译"成回答(response)
			
			
			
			2. 长对话模型 vs. 短对话模型
			
			短对话（Short Conversation）指的是一问一答式的单轮（single turn）对话。举例来说，当机器收到用户的一个提问时，会返回一个合适的回答。对应地，长对话（Long Conversation）指的是你来我往的多轮（multi-turn）对话，例如两个朋友对某个话题交流意见的一段聊天。在这个场景中，需要谈话双方（聊天机器人可能是其中一方）记得双方曾经谈论过什么，这是和短对话的场景的区别之一。现下，机器人客服系统通常是长对话模型
			
			3. 开放话题模型 vs. 封闭话题模型
			
			开放话题（Open Domain）场景下，用户可以说任何内容，不需要是有特定的目的或是意图的询问。人们在Twitter、Reddit等社交网络上的对话形式就是典型的开放话题情景。由于该场景下，可谈论的主题的数量不限，而且需要一些常识作为聊天基础，使得搭建一个这样的聊天机器人变得相对困难。
			封闭话题（Closed Domain）场景，又称为目标驱动型（goal-driven），系统致力于解决特定领域的问题，因此可能的询问和回答的数量相对有限。技术客服系统或是购物助手等应用就是封闭话题模型的例子。我们不要求这些系统能够谈论政治，只需要它们能够尽可能有效地解决我们的问题。虽然用户还是可以向这些系统问一些不着边际的问题，但是系统同样可以不着边际地给你回复 ;)
			
			Relevant Link:
			
			http://naturali.io/deeplearning/chatbot/introduction/2016/04/28/chatbot-part1.html
			http://blog.topspeedsnail.com/archives/10735/comment-page-1#comment-1161
			http://blog.csdn.net/malefactor/article/details/51901115
			 
			
			1. 训练语料库
			
			wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip
			解压
			unzip dgk_shooter_min.conv.zip
			Relevant Link:
			
			https://github.com/rustch3n/dgk_lost_conv
			 
			
			2. 数据预处理
			
			一般来说，我们拿到的基础语料库可能是一些电影台词对话，或者是UBUNTU对话语料库(Ubuntu Dialog Corpus)，但基本上我们都要完成以下几大步骤
			
			1. 分词(tokenized)
			2. 英文单词取词根(stemmed)
			3. 英文单词变形的归类(lemmatized)(例如单复数归类)等
			4. 此外，例如人名、地名、组织名、URL链接、系统路径等专有名词，我们也可以统一用类型标识符来替代 
			M 表示话语，E 表示分割，遇到M就吧当前对话片段加入临时对话集，遇到E就说明遇到一个中断或者交谈双方转换了，一口气吧临时对话集加入convs总对话集，一次加入一个对话集，可以理解为拍电影里面的一个"咔"
			
			复制代码
			convs = []  # conversation set
			with open(conv_path, encoding="utf8") as f:
			    one_conv = []  # a complete conversation
			    for line in f:
			        line = line.strip('\n').replace('/', '')
			        if line == '':
			            continue
			        if line[0] == 'E':
			            if one_conv:
			                convs.append(one_conv)
			            one_conv = []
			        elif line[0] == 'M':
			            one_conv.append(line.split(' ')[1])
			复制代码
			因为场景是聊天机器人，影视剧的台词也是一人一句对答的，所以这里需要忽略2种特殊情况，只有一问或者只有一答，以及问和答的数量不一致，即最后一个人问完了没有得到回答
			
			复制代码
			# Grasping calligraphy answer answer
			ask = []  # ask
			response = []  # answers
			for conv in convs:
			    if len(conv) == 1:
			        continue
			    if len(conv) % 2 != 0:
			        conv = conv[:-1]
			    for i in range(len(conv)):
			        if i % 2 == 0:
			            ask.append(conv[i])
			        else:
			            response.append(conv[i])
			复制代码
			
			
			
			
			Relevant Link:
			
			 
			
			3. 词汇转向量
			
			我们知道图像识别、语音识别之所以能率先在深度学习领域取得较大成就，其中一个原因在于这2个领域的原始输入数据本身就带有很强的样本关联性，例如像素权重分布在同一类物体的不同图像中，表现是基本一致的，这本质上也人脑识别同类物体的机制是一样的，即我们常说的"举一反三"能力，我们学过的文字越多，就越可能驾驭甚至能创造组合出新的文字用法，写出华丽的文章
			
			但是NPL或者语义识别领域的输入数据，对话或者叫语料往往是不具备这种强关联性的，为此，就需要引入一个概念模型，叫词向量(word2vec)或短语向量(seq2seq)，简单来说就是将语料库中的词汇抽象映射到一个向量空间中，向量的排布是根据预发和词义语境决定的，例如，"中国->人"(中国后面紧跟着一个人字的可能性是极大的)、"你今年几岁了->我 ** 岁了"
			
			0x1: Token化处理、词编码
			
			将训练集中的对话的每个文件拆分成单独的一个个文字，形成一个词表(word table)
			
			复制代码
			def gen_vocabulary_file(input_file, output_file):
			    vocabulary = {}
			    with open(input_file) as f:
			        counter = 0
			        for line in f:
			            counter += 1
			            tokens = [word for word in line.strip()]
			            for word in tokens:
			                if word in vocabulary:
			                    vocabulary[word] += 1
			                else:
			                    vocabulary[word] = 1
			        vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
			        # For taking 10000 custom character kanji
			        if len(vocabulary_list) > 10000:
			            vocabulary_list = vocabulary_list[:10000]
			        print(input_file + " phrase table size:", len(vocabulary_list))
			        with open(output_file, "w") as ff:
			            for word in vocabulary_list:
			                ff.write(word + "\n")
			复制代码
			
			
			完成了Token化之后，需要对单词进行数字编码，方便后续的向量空间处理，这里依据的核心思想是这样的
			
			我们的训练语料库的对话之间都是有强关联的，基于这份有关联的对话集获得的词表的词之间也有逻辑关联性，那么我们只要按照此表原生的顺序对词进行编码，这个编码后的[work, id]就是一个有向量空间关联性的词表
			
			复制代码
			def convert_conversation_to_vector(input_file, vocabulary_file, output_file):
			    tmp_vocab = []
			    with open(vocabulary_file, "r") as f:
			        tmp_vocab.extend(f.readlines())
			    tmp_vocab = [line.strip() for line in tmp_vocab]
			    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
			    for item in vocab:
			        print item.encode('utf-8')
			复制代码
			所以我们根据训练预料集得到的此表可以作为对话训练集和对话测试机进行向量化的依据，我们的目的是将对话(包括训练集和测试集)的问和答都转化映射到向量空间
			
			土 968
			"土"字在训练集词汇表中的位置是968，我们就给该字设置一个编码968
			0x2: 对话转为向量
			
			原作者在词表的选取上作了裁剪，只选取前5000个词汇，但是仔细思考了一下，感觉问题源头还是在训练语料库不够丰富，不能完全覆盖所有的对话语言场景
			
			
			
			这一步得到一个ask/answer的语句seq向量空间集，对于训练集，我们将ask和answer建立映射关系
			
			Relevant Link:
			
			 
			
			4. 训练
			
			0x1: Sequence-to-sequence basics
			
			A basic sequence-to-sequence model, as introduced in Cho et al., 2014, consists of two recurrent neural networks (RNNs): an encoder that processes the input and a decoder that generates the output. This basic architecture is depicted below.
			
			
			
			Each box in the picture above represents a cell of the RNN, most commonly a GRU cell or an LSTM cell. Encoder and decoder can share weights or, as is more common, use a different set of parameters. Multi-layer cells have been successfully used in sequence-to-sequence models too 
			In the basic model depicted above, every input has to be encoded into a fixed-size state vector, as that is the only thing passed to the decoder. To allow the decoder more direct access to the input, an attention mechanism was introduced in Bahdanau et al., 2014.; suffice it to say that it allows the decoder to peek into the input at every decoding step. A multi-layer sequence-to-sequence network with LSTM cells and attention mechanism in the decoder looks like this.
			
			
			
			0x2: 训练过程
			
			利用ask/answer的训练集输入神经网络，并使用ask/answer测试向量映射集实现BP反馈与，使用一个三层神经网络，让tensorflow自动调整权重参数，获得一个ask-?的模型
			
			复制代码
			# -*- coding: utf-8 -*-
			
			import tensorflow as tf  # 0.12
			from tensorflow.models.rnn.translate import seq2seq_model
			import os
			import numpy as np
			import math
			
			PAD_ID = 0
			GO_ID = 1
			EOS_ID = 2
			UNK_ID = 3
			
			# ask/answer conversation vector file
			train_ask_vec_file = 'train_ask.vec'
			train_answer_vec_file = 'train_answer.vec'
			test_ask_vec_file = 'test_ask.vec'
			test_answer_vec_file = 'test_answer.vec'
			
			# word table 6000
			vocabulary_ask_size = 6000
			vocabulary_answer_size = 6000
			
			buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
			layer_size = 256
			num_layers = 3
			batch_size = 64
			
			
			# read *dencode.vec和*decode.vec data into memory
			def read_data(source_path, target_path, max_size=None):
			    data_set = [[] for _ in buckets]
			    with tf.gfile.GFile(source_path, mode="r") as source_file:
			        with tf.gfile.GFile(target_path, mode="r") as target_file:
			            source, target = source_file.readline(), target_file.readline()
			            counter = 0
			            while source and target and (not max_size or counter < max_size):
			                counter += 1
			                source_ids = [int(x) for x in source.split()]
			                target_ids = [int(x) for x in target.split()]
			                target_ids.append(EOS_ID)
			                for bucket_id, (source_size, target_size) in enumerate(buckets):
			                    if len(source_ids) < source_size and len(target_ids) < target_size:
			                        data_set[bucket_id].append([source_ids, target_ids])
			                        break
			                source, target = source_file.readline(), target_file.readline()
			    return data_set
			
			if __name__ == '__main__':
			    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_ask_size,
			                                       target_vocab_size=vocabulary_answer_size,
			                                       buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
			                                       batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97,
			                                       forward_only=False)
			
			    config = tf.ConfigProto()
			    config.gpu_options.allocator_type = 'BFC'  # forbidden out of memory
			
			    with tf.Session(config=config) as sess:
			        # 恢复前一次训练
			        ckpt = tf.train.get_checkpoint_state('.')
			        if ckpt != None:
			            print(ckpt.model_checkpoint_path)
			            model.saver.restore(sess, ckpt.model_checkpoint_path)
			        else:
			            sess.run(tf.global_variables_initializer())
			
			        train_set = read_data(train_ask_vec_file, train_answer_vec_file)
			        test_set = read_data(test_ask_vec_file, test_answer_vec_file)
			
			        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
			        train_total_size = float(sum(train_bucket_sizes))
			        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
			
			        loss = 0.0
			        total_step = 0
			        previous_losses = []
			        # continue train，save modle after a decade of time
			        while True:
			            random_number_01 = np.random.random_sample()
			            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
			
			            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
			            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
			
			            loss += step_loss / 500
			            total_step += 1
			
			            print(total_step)
			            if total_step % 500 == 0:
			                print(model.global_step.eval(), model.learning_rate.eval(), loss)
			
			                # if model has't not improve，decrese the learning rate
			                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
			                    sess.run(model.learning_rate_decay_op)
			                previous_losses.append(loss)
			                # save model
			                checkpoint_path = "chatbot_seq2seq.ckpt"
			                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
			                loss = 0.0
			                # evaluation the model by test dataset
			                for bucket_id in range(len(buckets)):
			                    if len(test_set[bucket_id]) == 0:
			                        continue
			                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
			                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
			                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
			                    print(bucket_id, eval_ppx)
			复制代码
			Relevant Link:
			
			https://www.tensorflow.org/tutorials/seq2seq
			http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
			 
			
			5. 聊天机器人 - 验证效果
			
			复制代码
			# -*- coding: utf-8 -*-
			
			import tensorflow as tf  # 0.12
			from tensorflow.models.rnn.translate import seq2seq_model
			import os
			import sys
			import locale
			import numpy as np
			
			PAD_ID = 0
			GO_ID = 1
			EOS_ID = 2
			UNK_ID = 3
			
			train_ask_vocabulary_file = "train_ask_vocabulary.vec"
			train_answer_vocabulary_file = "train_answer_vocabulary.vec"
			
			
			def read_vocabulary(input_file):
			    tmp_vocab = []
			    with open(input_file, "r") as f:
			        tmp_vocab.extend(f.readlines())
			    tmp_vocab = [line.strip() for line in tmp_vocab]
			    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
			    return vocab, tmp_vocab
			
			
			if __name__ == '__main__':
			    vocab_en, _, = read_vocabulary(train_ask_vocabulary_file)
			    _, vocab_de, = read_vocabulary(train_answer_vocabulary_file)
			
			    # word table 6000
			    vocabulary_ask_size = 6000
			    vocabulary_answer_size = 6000
			
			    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
			    layer_size = 256
			    num_layers = 3
			    batch_size = 1
			
			    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_ask_size,
			                                       target_vocab_size=vocabulary_answer_size,
			                                       buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
			                                       batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.99,
			                                       forward_only=True)
			    model.batch_size = 1
			
			    with tf.Session() as sess:
			        # restore last train
			        ckpt = tf.train.get_checkpoint_state('.')
			        if ckpt != None:
			            print(ckpt.model_checkpoint_path)
			            model.saver.restore(sess, ckpt.model_checkpoint_path)
			        else:
			            print("model not found")
			
			        while True:
			            input_string = raw_input('me > ').decode(sys.stdin.encoding or locale.getpreferredencoding(True)).strip()
			            # 退出
			            if input_string == 'quit':
			                exit()
			
			            # convert the user's input to vector
			            input_string_vec = []
			            for words in input_string.strip():
			                input_string_vec.append(vocab_en.get(words, UNK_ID))
			            bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
			            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
			                                                                             bucket_id)
			            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
			            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
			            if EOS_ID in outputs:
			                outputs = outputs[:outputs.index(EOS_ID)]
			
			            response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
			            print('AI > ' + response)
			复制代码
			神经网络还是很依赖样本的训练的，我在实验的过程中发现，用GPU跑到20000 step之后，模型的效果才逐渐显现出来，才开始逐渐像正常的人机对话了
	
----------------------------

新手入门 完整教程 进阶指南 API中文手册 精华文章 TF社区 Fork me on GitHub
TensorFlow 是一个用于人工智能的开源神器

GET STARTED
关于 TensorFlow
TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用性使其也可广泛用于其他计算领域。
这次的开源发布版本支持单pc或单移动设备上的计算。

观看视频
Tensors Flowing
什么是数据流图（Data Flow Graph）?
数据流图用“结点”（nodes）和“线”(edges)的有向图来描述数学计算。“节点” 一般用来表示施加的数学操作，但也可以表示数据输入（feed in）的起点/输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。“线”表示“节点”之间的输入/输出关系。这些数据“线”可以输运“size可动态调整”的多维数据数组，即“张量”（tensor）。张量从图中流过的直观图像是这个工具取名为“Tensorflow”的原因。一旦输入端的所有张量准备好，节点将被分配到各种计算设备完成异步并行地执行运算。
TensorFlow的特征

高度的灵活性

TensorFlow 不是一个严格的“神经网络”库。只要你可以将你的计算表示为一个数据流图，你就可以使用Tensorflow。你来构建图，描写驱动计算的内部循环。我们提供了有用的工具来帮助你组装“子图”（常用于神经网络），当然用户也可以自己在Tensorflow基础上写自己的“上层库”。定义顺手好用的新复合操作和写一个python函数一样容易，而且也不用担心性能损耗。当然万一你发现找不到想要的底层数据操作，你也可以自己写一点c++代码来丰富底层的操作。
真正的可移植性（Portability）

Tensorflow 在CPU和GPU上运行，比如说可以运行在台式机、服务器、手机移动设备等等。想要在没有特殊硬件的前提下，在你的笔记本上跑一下机器学习的新想法？Tensorflow可以办到这点。准备将你的训练模型在多个CPU上规模化运算，又不想修改代码？Tensorflow可以办到这点。想要将你的训练好的模型作为产品的一部分用到手机app里？Tensorflow可以办到这点。你改变主意了，想要将你的模型作为云端服务运行在自己的服务器上，或者运行在Docker容器里？Tensorfow也能办到。Tensorflow就是这么拽 :)
将科研和产品联系在一起

过去如果要将科研中的机器学习想法用到产品中，需要大量的代码重写工作。那样的日子一去不复返了！在Google，科学家用Tensorflow尝试新的算法，产品团队则用Tensorflow来训练和使用计算模型，并直接提供给在线用户。使用Tensorflow可以让应用型研究者将想法迅速运用到产品中，也可以让学术性研究者更直接地彼此分享代码，从而提高科研产出率。
自动求微分

基于梯度的机器学习算法会受益于Tensorflow自动求微分的能力。作为Tensorflow用户，你只需要定义预测模型的结构，将这个结构和目标函数（objective function）结合在一起，并添加数据，Tensorflow将自动为你计算相关的微分导数。计算某个变量相对于其他变量的导数仅仅是通过扩展你的图来完成的，所以你能一直清楚看到究竟在发生什么。
多语言支持

Tensorflow 有一个合理的c++使用界面，也有一个易用的python使用界面来构建和执行你的graphs。你可以直接写python/c++程序，也可以用交互式的ipython界面来用Tensorflow尝试些想法，它可以帮你将笔记、代码、可视化等有条理地归置好。当然这仅仅是个起点——我们希望能鼓励你创造自己最喜欢的语言界面，比如Go，Java，Lua，Javascript，或者是R。
性能最优化

比如说你又一个32个CPU内核、4个GPU显卡的工作站，想要将你工作站的计算潜能全发挥出来？由于Tensorflow 给予了线程、队列、异步操作等以最佳的支持，Tensorflow 让你可以将你手边硬件的计算潜能全部发挥出来。你可以自由地将Tensorflow图中的计算元素分配到不同设备上，Tensorflow可以帮你管理好这些不同副本。
谁可以用 TensorFlow?
任何人都可以用Tensorflow。学生、研究员、爱好者、极客、工程师、开发者、发明家、创业者等等都可以在Apache 2.0 开源协议下使用Tensorflow。
Tensorflow 还没竣工，它需要被进一步扩展和上层建构。我们刚发布了源代码的最初版本，并且将持续完善它。我们希望大家通过直接向源代码贡献，或者提供反馈，来建立一个活跃的开源社区，以推动这个代码库的未来发展。
为啥Google要开源这个神器?
如果Tensorflow这么好，为啥不藏起来而是要开源呢？答案或许比你想象的简单：我们认为机器学习是未来新产品和新技术的一个关键部分。在这一个领域的研究是全球性的，并且发展很快，却缺少一个标准化的工具。通过分享这个我们认为是世界上最好的机器学习工具库之一的东东，我们希望能够创造一个开放的标准，来促进交流研究想法和将机器学习算法产品化。Google的工程师们确实在用它来提供用户直接在用的产品和服务，而Google的研究团队也将在他们的许多科研文章中分享他们对Tensorflow的使用。
开始试用TENSORFLOW

----------------------------

word2vec是一个将单词转换成向量形式的工具。可以把对文本内容的处理简化为向量空间中的向量运算，计算出向量空间上的相似度，来表示文本语义上的相似度。

一、理论概述

（主要来源于http://licstar.net/archives/328这篇博客）

1.词向量是什么

自然语言理解的问题要转化为机器学习的问题，第一步肯定是要找一种方法把这些符号数学化。

　　NLP 中最直观，也是到目前为止最常用的词表示方法是 One-hot Representation，这种方法把每个词表示为一个很长的向量。这个向量的维度是词表大小，其中绝大多数元素为 0，只有一个维度的值为 1，这个维度就代表了当前的词。

　　举个栗子，

　　“话筒”表示为 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ...]

　　“麦克”表示为 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ...]

　　每个词都是茫茫 0 海中的一个 1。

　　这种 One-hot Representation 如果采用稀疏方式存储，会是非常的简洁：也就是给每个词分配一个数字 ID。比如刚才的例子中，话筒记为 3，麦克记为 8（假设从 0 开始记）。如果要编程实现的话，用 Hash 表给每个词分配一个编号就可以了。这么简洁的表示方法配合上最大熵、SVM、CRF 等等算法已经很好地完成了 NLP 领域的各种主流任务。

　　当然这种表示方法也存在一个重要的问题就是“词汇鸿沟”现象：任意两个词之间都是孤立的。光从这两个向量中看不出两个词是否有关系，哪怕是话筒和麦克这样的同义词也不能幸免于难。

　　Deep Learning 中一般用到的词向量并不是刚才提到的用 One-hot Representation 表示的那种很长很长的词向量，而是用 Distributed Representation（不知道这个应该怎么翻译，因为还存在一种叫“Distributional Representation”(类似，LDA中用topic表示词语的词向量的表示方法）表示的一种低维实数向量。这种向量一般是这个样子：[0.792, −0.177, −0.107, 0.109, −0.542, ...]。维度以 50 维和 100 维比较常见。

2.词向量的来历

Distributed representation 最早是 Hinton 在 1986 年的论文《Learning distributed representations of concepts》中提出的。虽然这篇文章没有说要将词做 Distributed representation但至少这种先进的思想在那个时候就在人们的心中埋下了火种，到 2000 年之后开始逐渐被人重视。

3. 词向量的训练

　　要介绍词向量是怎么训练得到的，就不得不提到语言模型。到目前为止我了解到的所有训练方法都是在训练语言模型的同时，顺便得到词向量的。

　　这也比较容易理解，要从一段无标注的自然文本中学习出一些东西，无非就是统计出词频、词的共现、词的搭配之类的信息。而要从自然文本中统计并建立一个语言模型，无疑是要求最为精确的一个任务（也不排除以后有人创造出更好更有用的方法）。既然构建语言模型这一任务要求这么高，其中必然也需要对语言进行更精细的统计和分析，同时也会需要更好的模型，更大的数据来支撑。目前最好的词向量都来自于此，也就不难理解了。

　　　　词向量的训练最经典的有 3 个工作，C&W 2008、M&H 2008、Mikolov 2010。当然在说这些工作之前，不得不介绍一下这一系列中 Bengio 的经典之作

4. 词向量的评价

 

词向量的评价大体上可以分成两种方式，第一种是把词向量融入现有系统中，看对系统性能的提升；第二种是直接从语言学的角度对词向量进行分析，如相似度、语义偏移等。

 

4.1 提升现有系统

　　词向量的用法最常见的有两种：

　　1. 直接用于神经网络模型的输入层。如 C&W 的 SENNA 系统中，将训练好的词向量作为输入，用前馈网络和卷积网络完成了词性标注、语义角色标注等一系列任务。再如 Socher 将词向量作为输入，用递归神经网络完成了句法分析、情感分析等多项任务。

2. 作为辅助特征扩充现有模型。如 Turian 将词向量作为额外的特征加入到接近 state of the art 的方法中，进一步提高了命名实体识别和短语识别的效果。

4.2 语言学评价

　　还有一个有意思的分析是 Mikolov 在 2013 年刚刚发表的一项发现。他发现两个词向量之间的关系，可以直接从这两个向量的差里体现出来。向量的差就是数学上的定义，直接逐位相减。比如 C(king)−C(queen)≈C(man)−C(woman)。更强大的是，与 C(king)−C(man)+C(woman) 最接近的向量就是 C(queen)。

　　为了分析词向量的这个特点， Mikolov 使用类比（analogy）的方式来评测。如已知 a 之于 b 犹如 c 之于 d。现在给出 a、b、c，看 C(a)−C(b)+C(c) 最接近的词是否是 d。

　　在文章 Mikolov 对比了词法关系（名词单复数 good-better:rough-rougher、动词第三人称单数、形容词比较级最高级等）和语义关系（clothing-shirt:dish-bowl）

 

这些实验结果中最容易理解的是：语料越大，词向量就越好。其它的实验由于缺乏严格控制条件进行对比，谈不上哪个更好哪个更差。不过这里的两个语言学分析都非常有意思，尤其是向量之间存在这种线性平移的关系，可能会是词向量发展的一个突破口。

<!--[if !vml]--><!--[endif]-->

 

关于Deep Lerning In Nlp的一些相关论文，《Deep Learning in NLP （一）词向量和语言模型》（http://licstar.net/archives/328）这篇博客总结的非常的好。以上内容大多数都是截取原博客内容。

 

二、实际操作

这篇文章是最近几天看word2vec源码以及相关神经网络训练词向量论文之后的个人小小的总结，主要是针对word2vec的使用，做一下介绍。望大家使用的过程中，少走弯路。

word2vec工具中包含了对两种模型的训练，如下图。在训练每种模型的时候又分HS和NEG两种方法。(看图就可以发现，其实word2vec并不deep……)

 

<!--[if !vml]--><!--[endif]-->

除了google自己的word2vec工具，各位对词向量感兴趣的牛人们也相继编写了各自不同的版本。其中比较好用的是Python Gensim主题模型包中的word2vec,但通过阅读其源码python版本只实现了skip-gram模型，并且只实现了通过分层softmax方法对其训练，并没有使用negative sampling。下面列举一下目前出现的版本以及相对应的地址，供大家选择。如下表：

 

----------------------------


## Anaconda概述

Anaconda是一个用于科学计算的Python发行版，支持 Linux, Mac, Windows系统，提供了包管理与环境管理的功能，可以很方便地解决多版本python并存、切换以及各种第三方包安装问题。Anaconda利用工具/命令conda来进行package和environment的管理，并且已经包含了Python和相关的配套工具。

这里先解释下conda、anaconda这些概念的差别。conda可以理解为一个工具，也是一个可执行命令，其核心功能是包管理与环境管理。包管理与pip的使用类似，环境管理则允许用户方便地安装不同版本的python并可以快速切换。Anaconda则是一个打包的集合，里面预装好了conda、某个版本的python、众多packages、科学计算工具等等，所以也称为Python的一种发行版。其实还有Miniconda，顾名思义，它只包含最基本的内容——python与conda，以及相关的必须依赖项，对于空间要求严格的用户，Miniconda是一种选择。

进入下文之前，说明一下conda的设计理念——conda将几乎所有的工具、第三方包都当做package对待，甚至包括python和conda自身！因此，conda打破了包管理与环境管理的约束，能非常方便地安装各种版本python、各种package并方便地切换。

### Anaconda的安装

Anaconda的下载页参见官网下载，Linux、Mac、Windows均支持。

安装时，会发现有两个不同版本的Anaconda，分别对应Python 2.7和Python 3.5，两个版本其实除了这点区别外其他都一样。后面我们会看到，安装哪个版本并不本质，因为通过环境管理，我们可以很方便地切换运行时的Python版本。（由于我常用的Python是2.7和3.4，因此倾向于直接安装Python 2.7对应的Anaconda）

下载后直接按照说明安装即可。这里想提醒一点：尽量按照Anaconda默认的行为安装——不使用root权限，仅为个人安装，安装目录设置在个人主目录下（Windows就无所谓了）。这样的好处是，同一台机器上的不同用户完全可以安装、配置自己的Anaconda，不会互相影响。

对于Mac、Linux系统，Anaconda安装好后，实际上就是在主目录下多了个文件夹（~/anaconda）而已，Windows会写入注册表。安装时，安装程序会把bin目录加入PATH（Linux/Mac写入~/.bashrc，Windows添加到系统变量PATH），这些操作也完全可以自己完成。以Linux/Mac为例，安装完成后设置PATH的操作是

	
	# 将anaconda的bin目录加入PATH，根据版本不同，也可能是~/anaconda3/bin
	echo 'export PATH="~/anaconda2/bin:$PATH"' >> ~/.bashrc
	# 更新bashrc以立即生效
	source ~/.bashrc
	
	# 将anaconda的bin目录加入PATH，根据版本不同，也可能是~/anaconda3/bin
	echo 'export PATH="~/anaconda2/bin:$PATH"' >> ~/.bashrc
	# 更新bashrc以立即生效
	source ~/.bashrc

配置好PATH后，可以通过which conda或conda --version命令检查是否正确。假如安装的是Python 2.7对应的版本，运行python --version或python -V可以得到Python 2.7.12 :: Anaconda 4.1.1 (64-bit)，也说明该发行版默认的环境是Python 2.7。

### Conda的环境管理

Conda的环境管理功能允许我们同时安装若干不同版本的Python，并能自由切换。对于上述安装过程，假设我们采用的是Python 2.7对应的安装包，那么Python 2.7就是默认的环境（默认名字是root，注意这个root不是超级管理员的意思）。

假设我们需要安装Python 3.4，此时，我们需要做的操作如下：


	# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）
	conda create --name python34 python=3.4
	
	# 安装好后，使用activate激活某个环境
	activate python34 # for Windows
	source activate python34 # for Linux & Mac
	# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH
	
	# 此时，再次输入
	python --version
	# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境
	
	# 如果想返回默认的python 2.7环境，运行
	deactivate python34 # for Windows
	source deactivate python34 # for Linux & Mac
	
	# 删除一个已有的环境
	conda remove --name python34 --all
	# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）
	conda create --name python34 python=3.4
	 
	# 安装好后，使用activate激活某个环境
	activate python34 # for Windows
	source activate python34 # for Linux & Mac
	# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH
	 
	# 此时，再次输入
	python --version
	# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境
	 
	# 如果想返回默认的python 2.7环境，运行
	deactivate python34 # for Windows
	source deactivate python34 # for Linux & Mac
	 
	# 删除一个已有的环境
	conda remove --name python34 --all

用户安装的不同python环境都会被放在目录~/anaconda/envs下，可以在命令中运行conda info -e查看已安装的环境，当前被激活的环境会显示有一个星号或者括号。

说明：有些用户可能经常使用python 3.4环境，因此直接把~/anaconda/envs/python34下面的bin或者Scripts加入PATH，去除anaconda对应的那个bin目录。这个办法，怎么说呢，也是可以的，但总觉得不是那么elegant……

如果直接按上面说的这么改PATH，你会发现conda命令又找不到了（当然找不到啦，因为conda在~/anaconda/bin里呢），这时候怎么办呢？方法有二：1. 显式地给出conda的绝对地址 2. 在python34环境中也安装conda工具（推荐）。

### Conda的包管理

Conda的包管理就比较好理解了，这部分功能与pip类似。

例如，如果需要安装scipy：


	# 安装scipy
	conda install scipy
	# conda会从从远程搜索scipy的相关信息和依赖项目，对于python 3.4，conda会同时安装numpy和mkl（运算加速的库）
	
	# 查看已经安装的packages
	conda list
	# 最新版的conda是从site-packages文件夹中搜索已经安装的包，不依赖于pip，因此可以显示出通过各种方式安装的包
	# 安装scipy
	conda install scipy
	# conda会从从远程搜索scipy的相关信息和依赖项目，对于python 3.4，conda会同时安装numpy和mkl（运算加速的库）
	 
	# 查看已经安装的packages
	conda list
	# 最新版的conda是从site-packages文件夹中搜索已经安装的包，不依赖于pip，因此可以显示出通过各种方式安装的包
	conda的一些常用操作如下：
	
	
	# 查看当前环境下已安装的包
	conda list
	
	# 查看某个指定环境的已安装包
	conda list -n python34
	
	# 查找package信息
	conda search numpy
	
	# 安装package
	conda install -n python34 numpy
	# 如果不用-n指定环境名称，则被安装在当前活跃环境
	# 也可以通过-c指定通过某个channel安装
	
	# 更新package
	conda update -n python34 numpy
	
	# 删除package
	conda remove -n python34 numpy
	
	# 查看当前环境下已安装的包
	conda list
	 
	# 查看某个指定环境的已安装包
	conda list -n python34
	 
	# 查找package信息
	conda search numpy
	 
	# 安装package
	conda install -n python34 numpy
	# 如果不用-n指定环境名称，则被安装在当前活跃环境
	# 也可以通过-c指定通过某个channel安装
	 
	# 更新package
	conda update -n python34 numpy
	 
	# 删除package
	conda remove -n python34 numpy
	前面已经提到，conda将conda、python等都视为package，因此，完全可以使用conda来管理conda和python的版本，例如
	
	
	# 更新conda，保持conda最新
	conda update conda
	
	# 更新anaconda
	conda update anaconda
	
	# 更新python
	conda update python
	# 假设当前环境是python 3.4, conda会将python升级为3.4.x系列的当前最新版本
	
	# 更新conda，保持conda最新
	conda update conda
	 
	# 更新anaconda
	conda update anaconda
	 
	# 更新python
	conda update python
	# 假设当前环境是python 3.4, conda会将python升级为3.4.x系列的当前最新版本
	补充：如果创建新的python环境，比如3.4，运行conda create -n python34 python=3.4之后，conda仅安装python 3.4相关的必须项，如python, pip等，如果希望该环境像默认环境那样，安装anaconda集合包，只需要：
	
	
	# 在当前环境下安装anaconda包集合
	conda install anaconda
	
	# 结合创建环境的命令，以上操作可以合并为
	conda create -n python34 python=3.4 anaconda
	# 也可以不用全部安装，根据需求安装自己需要的package即可
	
	# 在当前环境下安装anaconda包集合
	conda install anaconda
	 
	# 结合创建环境的命令，以上操作可以合并为
	conda create -n python34 python=3.4 anaconda
	# 也可以不用全部安装，根据需求安装自己需要的package即可
	设置国内镜像
	
	如果需要安装很多packages，你会发现conda下载的速度经常很慢，因为Anaconda.org的服务器在国外。所幸的是，清华TUNA镜像源有Anaconda仓库的镜像，我们将其加入conda的配置即可：
	
	
	# 添加Anaconda的TUNA镜像
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
	# TUNA的help中镜像地址加有引号，需要去掉
	
	# 设置搜索时显示通道地址
	conda config --set show_channel_urls yes
	
	# 添加Anaconda的TUNA镜像
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
	# TUNA的help中镜像地址加有引号，需要去掉
	 
	# 设置搜索时显示通道地址
	conda config --set show_channel_urls yes

执行完上述命令后，会生成~/.condarc(Linux/Mac)或C:UsersUSER_NAME.condarc文件，记录着我们对conda的配置，直接手动创建、编辑该文件是相同的效果。

跋

Anaconda具有跨平台、包管理、环境管理的特点，因此很适合快速在新的机器上部署Python环境。总结而言，整套安装、配置流程如下：

### 下载Anaconda、安装

配置PATH（bashrc或环境变量），更改TUNA镜像源
创建所需的不用版本的python环境
Just Try!
cheat-sheet 下载：
	Conda cheat sheet


----------------------------

Theano 

-----------------------------

# tensorflow

TensorFlow核心使用技巧

为了介绍TensorFlow的各种用法，我们将使用deep_recommend_system这个开源项目，它实现了TFRecords、QueueRunner、Checkpoint、TensorBoard、Inference、GPU支持、分布式训练和多层神经网络模型等特性，而且可以轻易拓展实现Wide and deep等模型，在实际的项目开发中可以直接下载使用。



1. 准备训练数据

一般TensorFlow应用代码包含Graph的定义和Session的运行，代码量不大可以封装到一个文件中，如cancer_classifier.py文件。训练前需要准备样本数据和测试数据，一般数据文件是空格或者逗号分隔的CSV文件，但TensorFlow建议使用二进制的TFRecords格式，这样可以支持QueuRunner和Coordinator进行多线程数据读取，并且可以通过batch size和epoch参数来控制训练时单次batch的大小和对样本文件迭代训练多少轮。如果直接读取CSV文件，需要在代码中记录下一次读取数据的指针，而且在样本无法全部加载到内存时使用非常不便。

在data目录，项目已经提供了CSV与TFRecords格式转换工具convert_cancer_to_tfrecords.py，参考这个脚本你就可以parse任意格式的CSV文件，转成TensorFlow支持的TFRecords格式。无论是大数据还是小数据，通过简单的脚本工具就可以直接对接TensorFlow，项目中还提供print_cancer_tfrecords.py脚本来调用API直接读取TFRecords文件的内容。



2. 接受命令行参数

有了TFRecords，我们就可以编写代码来训练神经网络模型了，但众所周知，深度学习有过多的Hyperparameter需要调优，我们就优化算法、模型层数和不同模型都需要不断调整，这时候使用命令行参数是非常方便的。

TensorFlow底层使用了python-gflags项目，然后封装成tf.app.flags接口，使用起来非常简单和直观，在实际项目中一般会提前定义命令行参数，尤其在后面将会提到的Cloud Machine Learning服务中，通过参数来简化Hyperparameter的调优。



3. 定义神经网络模型

准备完数据和参数，最重要的还是要定义好网络模型，定义模型参数可以很简单，创建多个Variable即可，也可以做得比较复杂，例如使用使用tf.variable_scope()和tf.get_variables()接口。为了保证每个Variable都有独特的名字，而且能都轻易地修改隐层节点数和网络层数，我们建议参考项目中的代码，尤其在定义Variables时注意要绑定CPU，TensorFlow默认使用GPU可能导致参数更新过慢。



上述代码在生产环境也十分常见，无论是训练、实现inference还是验证模型正确率和auc时都会用到。项目中还基于此代码实现了Wide and deep模型，在Google Play应用商店的推荐业务有广泛应用，这也是适用于普遍的推荐系统，将传统的逻辑回归模型和深度学习的神经网络模型有机结合在一起。

4. 使用不同的优化算法

定义好网络模型，我们需要觉得使用哪种Optimizer去优化模型参数，是应该选择Sgd、Rmsprop还是选择Adagrad、Ftrl呢？对于不同场景和数据集没有固定的答案，最好的方式就是实践，通过前面定义的命令行参数我们可以很方便得使用不同优化算法来训练模型。



在生产实践中，不同优化算法在训练结果、训练速度上都有很大差异，过度优化网络参数可能效果没有使用其他优化算法来得有效，因此选用正确的优化算法也是Hyperparameter调优中很重要的一步，通过在TensorFlow代码中加入这段逻辑也可以很好地实现对应的功能。

5. Online learning与Continuous learning

很多机器学习厂商都会宣称自己的产品支持Online learning，其实这只是TensorFlow的一个基本的功能，就是支持在线数据不断优化模型。TensorFlow可以通过tf.train.Saver()来保存模型和恢复模型参数，使用Python加载模型文件后，可不断接受在线请求的数据，更新模型参数后通过Saver保存成checkpoint，用于下一次优化或者线上服务。



而Continuous training是指训练即使被中断，也能继续上一次的训练结果继续优化模型，在TensorFlow中也是通过Saver和checkpoint文件来实现。在deep_recommend_system项目默认能从上一次训练中继续优化模型，也可以在命令行中指定train_from_scratch，不仅不用担心训练进程被中断，也可以一边训练一边做inference提供线上服务。

6. 使用TensorBoard优化参数

TensorFlow还集成了一个功能强大的图形化工具，也即是TensorBoard，一般只需要在代码中加入我们关心的训练指标，TensorBoard就会自动根据这些参数绘图，通过可视化的方式来了解模型训练的情况。

tf.scalar_summary(‘loss’, loss)
tf.scalar_summary(‘accuracy’, accuracy)
tf.scalar_summary(‘auc’, auc_op)



7. 分布式TensorFlow应用

最后不得不介绍TensorFlow强大的分布式计算功能，传统的计算框架如Caffe，原生不支持分布式训练，在数据量巨大的情况下往往无法通过增加机器scale out。TensorFlow承载了Google各个业务PB级的数据，在设计之初就考虑到分布式计算的需求，通过gRPC、Protobuf等高性能库实现了神经网络模型的分布式计算。

实现分布式TensorFlow应用并不难，构建Graph代码与单机版相同，我们实现了一个分布式的cancer_classifier.py例子，通过下面的命令就可以启动多ps多worker的训练集群。

 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=1
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1
在深入阅读代码前，我们需要了解分布式TensorFlow中ps、worker、in-graph、between-graph、synchronous training和asynchronous training的概念。首先ps是整个训练集群的参数服务器，保存模型的Variable，worker是计算模型梯度的节点，得到的梯度向量会交付给ps更新模型。in-graph与between-graph对应，但两者都可以实现同步训练和异步训练，in-graph指整个集群由一个client来构建graph，并且由这个client来提交graph到集群中，其他worker只负责处理梯度计算的任务，而between-graph指的是一个集群中多个worker可以创建多个graph，但由于worker运行的代码相同因此构建的graph也相同，并且参数都保存到相同的ps中保证训练同一个模型，这样多个worker都可以构建graph和读取训练数据，适合大数据场景。同步训练和异步训练差异在于，同步训练每次更新梯度需要阻塞等待所有worker的结果，而异步训练不会有阻塞，训练的效率更高，在大数据和分布式的场景下一般使用异步训练。

8. Cloud Machine Learning

前面已经介绍了TensorFlow相关的全部内容，细心的网友可能已经发现，TensorFlow功能强大，但究其本质还是一个library，用户除了编写TensorFlow应用代码还需要在物理机上起服务，并且手动指定训练数据和模型文件的目录，维护成本比较大，而且机器之间不可共享。

纵观大数据处理和资源调度行业，Hadoop生态俨然成为了业界的标准，通过MapReduce或Spark接口来处理数据，用户通过API提交任务后由Yarn进行统一的资源分配和调度，不仅让分布式计算成为可能，也通过资源共享和统一调度平的台极大地提高了服务器的利用率。很遗憾TensorFlow定义是深度学习框架，并不包含集群资源管理等功能，但开源TensorFlow以后，Google很快公布了Google Cloud ML服务，我们从Alpha版本开始已经是Cloud ML的早期用户，深深体会到云端训练深度学习的便利性。通过Google Cloud ML服务，我们可以把TensorFlow应用代码直接提交到云端运行，甚至可以把训练好的模型直接部署在云上，通过API就可以直接访问，也得益于TensorFlow良好的设计，我们基于Kubernetes和TensorFlow serving实现了Cloud Machine Learning服务，架构设计和使用接口都与Google Cloud ML类似。



TensorFlow是很好深度学习框架，对于个人开发者、科研人员已经企业都是值得投资的技术方向，而Cloud Machine Learning可以解决用户在环境初始化、训练任务管理以及神经网络模型的在线服务上的管理和调度问题。目前Google Cloud ML已经支持automatically hyperparameter tunning，参数调优未来也将成为计算问题而不是技术问题，即使有的开发者使用MXNet或者其他，而不是TensorFlow，我们也愿意与更多深度学习用户和平台开发者交流，促进社区的发展。

最后总结

总结一下，本文主要介绍TensorFlow深度学习框架的学习与应用，通过deep_recommend_system项目介绍了下面使用TensorFlow的8个核心要点，也欢迎大家下载源码试用和反馈。

1. 准备训练数据

2. 接受命令行参数

3. 定义神经网络模型

4. 使用不同的优化算法

5. Online learning与Continuous learning

6. 使用TensorBoard优化参数

7. 分布式TensorFlow应用

8. Cloud Machine Learning

-----------------------------
神经元  Y=WX+B  ，通过输入的参数X ===========》Y 深度学习 每一个batch来说 其实就是 多项公式

数学里面 求多项公式 其实 就是 矩阵 W 矩阵 乘与 X  加上 B 矩阵 = Y矩阵 ，矩阵 二元数组在tensorflow 也是一个tensor ndarray , 通常 我们知道 因为relu 收敛效果要比sigmod 与tanh 要好，所以在cnn中常用relu，所以 其实 对于输出o=relu(wx+b) , 

----------------------------------

# 卷积
卷积是分析数学中一种重要的运算。设f（x）， g（x）是R1上的两个可积函数，作积分：
可以证明，关于几乎所有的x∈（－∞，∞） ，上述积分是存在的。这样，随着x的不同取值 ，这个积分就定义了一个新函数h(x)，称为f与g的卷积，记为h（x）=（f *g）（x）。容易验证，（f *g）（x）=（g *f）（x），并且（f *g）（x）仍为可积函数。这就是说，把卷积代替乘法，L1（R1）1空间是一个代数，甚至是巴拿赫代数。

卷积与傅里叶变换有着密切的关系。以(x) ，(x)表示L1（R）1中f和g的傅里叶变换，那么有如下的关系成立：（f *g）∧（x）=(x)·(x)，即两函数的傅里叶变换的乘积等于它们卷积后的傅里叶变换。这个关系，使傅里叶分析中许多问题的处理得到简化。

由卷积得到的函数（f *g）（x），一般要比f，g都光滑。特别当g为具有紧支集的光滑函数，f 为局部可积时，它们的卷积（f *g）（x）也是光滑函数。利用这一性质，对于任意的可积函数 ， 都可以简单地构造出一列逼近于f 的光滑函数列fs（x），这种方法称为函数的光滑化或正则化。
卷积的概念还可以推广到数列 、测度以及广义函数上去

卷积的重要的物理意义是：一个函数（如：单位响应）在另一个函数（如：输入信号）上的加权叠加。

对于线性时不变系统，如果知道该系统的单位响应，那么将单位响应和输入信号求卷积，就相当于把输入信号的各个时间点的单位响应 加权叠加，就直接得到了输出信号。

### 卷积神经网络

卷积神经网络（Convolutional Neural Network,CNN）是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。 它包括卷积层(alternating convolutional layer)和池层(pooling layer)。

### 为什么卷积

在图像处理中，往往把图像表示为像素的向量，比如一个1000×1000的图像，可以表示为一个1000000的向量。在神经网络中，如果隐含层数目与输入层一样，即也是1000000时，那么输入层到隐含层的参数数据为1000000×1000000=10^12，这样就太多了，基本没法训练。所以图像处理要想练成神经网络大法，必先减少参数加快速度。

### 卷积 

卷积神经网络有两种神器可以降低参数数目

- 局部感知野。
	- 一般认为人对外界的认知是从局部到全局的，而图像的空间联系也是局部的像素联系较为紧密，而距离较远的像素相关性则较弱。因而，每个神经元其实没有必要对全局图像进行感知，只需要对局部进行感知，然后在更高层将局部的信息综合起来就得到了全局的信息。网络部分连通的思想，也是受启发于生物学里面的视觉系统结构。视觉皮层的神经元就是局部接受信息的（即这些神经元只响应某些特定区域的刺激）

- 参数共享
	- 仍然过多，那么就启动第二级神器，即权值共享。在上面的局部连接中，每个神经元都对应100个参数，一共1000000个神经元，如果这1000000个神经元的100个参数都是相等的，那么参数数目就变为100了。怎么理解权值共享呢？我们可以这100个参数（也就是卷积操作）看成是提取特征的方式，该方式与位置无关。这其中隐含的原理则是：图像的一部分的统计特性与其他部分是一样的。这也意味着我们在这一部分学习的特征也能用在另一部分上，所以对于这个图像上的所有位置，我们都能使用同样的学习特征。更直观一些，当从一个大尺寸图像中随机选取一小块，比如说 8x8 作为样本，并且从这个小块样本中学习到了一些特征，这时我们可以把从这个 8x8 样本中学习到的特征作为探测器，应用到这个图像的任意地方中去。特别是，我们可以用从 8x8 样本中所学习到的特征跟原本的大尺寸图像作卷积，从而对这个大尺寸图像上的任一位置获得一个不同特征的激活值。如下图所示，展示了一个3×3的卷积核在5×5的图像上做卷积的过程。每个卷积都是一种特征提取方式，就像一个筛子，将图像中符合条件（激活值越大越符合条件）的部分筛选出来。


### 多核卷积

### Down-pooling 池化

在通过卷积获得了特征 (features) 之后，下一步我们希望利用这些特征去做分类。理论上讲，人们可以用所有提取得到的特征去训练分类器，例如 softmax 分类器，但这样做面临计算量的挑战。例如：对于一个 96X96 像素的图像，假设我们已经学习得到了400个定义在8X8输入上的特征，每一个特征和图像卷积都会得到一个 (96 − 8 + 1) × (96 − 8 + 1) = 7921 维的卷积特征，由于有 400 个特征，所以每个样例 (example) 都会得到一个 7921 × 400 = 3,168,400 维的卷积特征向量。学习一个拥有超过 3 百万特征输入的分类器十分不便，并且容易出现过拟合 (over-fitting)。

为了解决这个问题，首先回忆一下，我们之所以决定使用卷积后的特征是因为图像具有一种“静态性”的属性，这也就意味着在一个图像区域有用的特征极有可能在另一个区域同样适用。因此，为了描述大的图像，一个很自然的想法就是对不同位置的特征进行聚合统计，例如，人们可以计算图像一个区域上的某个特定特征的平均值 (或最大值)。这些概要统计特征不仅具有低得多的维度 (相比使用所有提取得到的特征)，同时还会改善结果(不容易过拟合)。

这种聚合的操作就叫做池化 (pooling)，有时也称为平均池化或者最大池化 (取决于计算池化的方法)。

### 多层卷积

在实际应用中，往往使用多层卷积，然后再使用全连接层进行训练，多层卷积的目的是一层卷积学到的特征往往是局部的，层数越高，学到的特征就越全局化。

[http://blog.csdn.net/stdcoutzyx/article/details/41596663/](http://http://blog.csdn.net/stdcoutzyx/article/details/41596663/)



----------------------------------

## 激活函数

### 作用

### 性质

激活函数通常有如下一些性质：

- 非线性： 当激活函数是线性的时候，一个两层的神经网络就可以逼近基本上所有的函数了。但是，如果激活函数是恒等激活函数的时候（即f(x)=x），就不满足这个性质了，而且如果MLP使用的是恒等激活函数，那么其实整个网络跟单层神经网络是等价的
- 可微性： 当优化方法是基于梯度的时候，这个性质是必须的。
- 单调性： 当激活函数是单调的时候，单层网络能够保证是凸函数。
- f(x)≈x： 当激活函数满足这个性质的时候，如果参数的初始化是random的很小的值，那么神经网络的训练将会很高效；如果不满足这个性质，那么就需要很用心的去设置初始值
- 输出值的范围： 当激活函数输出值是 有限 的时候，基于梯度的优化方法会更加 稳定，因为特征的表示受有限权值的影响更显著；当激活函数的输出是 无限 的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的learning rate.

### 怎么选择激活函数呢？

形象的说：激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。

- 有些数据是线性可分的，意思是，可以用一条直线将数据分开，这时候你需要通过一定的机器学习的方法，比如感知机算法(perceptron learning algorithm) 找到一个合适的线性方程。
- 有些数据不是线性可分的，第一个办法，是做线性变换(linear transformation)，比如讲x,y变成x^2,y^2，这样可以画出圆形，**如果将坐标轴从x,y变为以x^2,y^2为标准，你会发现数据经过变换后是线性可分的了**
- 第二种处理不是线性可分数据的方法是引入非线性模型，例如，异或问题(xor problem)。可以设计一种神经网络，通过激活函数使得数据线性可分。

[https://www.zhihu.com/question/22334626/answer/21036590](https://www.zhihu.com/question/22334626/answer/21036590)

如果你使用 ReLU，那么一定要小心设置 learning rate，而且要注意不要让你的网络出现很多 “dead” 神经元，如果这个问题不好解决，那么可以试试 Leaky ReLU、PReLU 或者 Maxout.

友情提醒：最好不要用 sigmoid，你可以试试 tanh，不过可以预期它的效果会比不上 ReLU 和 Maxout.

还有，通常来说，很少会把各种激活函数串起来在一个网络中使用的。