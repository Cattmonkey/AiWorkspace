#coding:utf-8

import requests

def main():
    r = requests.post('http://localhost/wx', data='''<xml>
 <ToUserName>to</ToUserName>
 <FromUserName>from</FromUserName>
 <CreateTime>1348831860</CreateTime>
 <MsgType>text</MsgType>
 <Content>hello</Content>
 <MsgId>1234567890123456</MsgId>
 </xml>''')
    print 'status:', r.status_code
    print 'text:', r.text

    r = requests.post('http://localhost/wx', data='''<xml>
 <ToUserName>from</ToUserName>
 <FromUserName>to</FromUserName>
 <CreateTime>1348831860</CreateTime>
 <MsgType>image</MsgType>
 <PicUrl>http://mmbiz.qpic.cn/mmbiz/85p6GTVdntaYYscTGZH4Akh5xZFj46CgoC40bxOiaFia0rA8YLcqicibwg1vu7ArEhicNwp4rOVmhxt8wU05NDibwuHA/0</PicUrl>
 <MediaId>2345</MediaId>
 <MsgId>1234567890123456</MsgId>
 </xml>''')
    print 'status:', r.status_code
    print 'text:', r.text

if __name__ == '__main__':
    main()
