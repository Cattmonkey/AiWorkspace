# -*- coding: utf-8 -*-
import tornado.escape
import tornado.web

from BaseHTTPServer import HTTPServer
import base64
import sys
import json
import time

import config
import time
import hashlib
import urllib
import os
import random

from util.xml2json import *

def wx_proc_msg(msg_body):
    try:
        jmsg = xml2json(msg_body)
        msgjson = json.loads(jmsg)
        msg = msgjson['xml']
        MsgType = msg['MsgType']

        main_content = {}
        main_content['MsgType'] = msg['MsgType']
        main_content['CreateTime'] = msg['CreateTime']
        main_content['ToUserName'] = msg['FromUserName']
        main_content['FromUserName'] = msg['ToUserName']

        if MsgType == 'text':
            req = msg['Content']
            respond = config.alice.respond(req)
            if respond == None or len(respond) < 1:
                respond = '''Sorry I can't understand you'''
            main_content['Content'] = respond
            result = {}
            result['xml'] = main_content.copy()

            result = json2xml(result)
            return result

        elif MsgType == 'image':
            main_content['MsgType'] = 'text'
            main_content['Content'] = '''Sorry I can't read picture.'''
            result = {}
            result['xml'] = main_content

            return json2xml(result)

        elif MsgType == 'voice':
            pass
        elif MsgType == 'video':
            pass
        elif MsgType == 'shortvideo':
            pass
        elif MsgType == 'location':
            pass
        elif MsgType == 'link':
            pass
        else:
            pass

    except Exception, e:
        print 'Error when process this message:', msg_body
        print e
    return ''

def check_wx_request(signature, timestamp, nonce):
    token = config.settings['wx_token']
    arr = [token, timestamp, nonce]
    arr.sort()
    sh = hashlib.sha1(arr[0] + arr[1] + arr[2]).hexdigest()
    if sh == signature:
        return True
    else:
        return False

class WX(tornado.web.RequestHandler):
    def get(self):
        signature = self.get_argument('signature', 'default')
        timestamp = self.get_argument('timestamp', 'default')
        nonce = self.get_argument('nonce', 'default')
        echostr = self.get_argument('echostr', 'default')
        if config.settings['wx_test'] or (signature != 'default' and timestamp != 'default' and nonce != 'default' and echostr != 'default' and check_wx_request(signature, timestamp, nonce)):
            self.write(echostr)
        else:
            self.write('Not Open')

    def post(self):
        signature = self.get_argument('signature', 'default')
        timestamp = self.get_argument('timestamp', 'default')
        nonce = self.get_argument('nonce', 'default')
        if config.settings['wx_test'] or (signature != 'default' and timestamp != 'default' and nonce != 'default' and check_wx_request(signature, timestamp, nonce)):
            body = self.request.body
            try:
                self.write(wx_proc_msg(body))
            except IOError, e:
                return
