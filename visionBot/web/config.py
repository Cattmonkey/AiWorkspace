# -*- coding: utf-8 -*-
import os
import tornado.web

settings = {
            'static_path': os.path.join(os.path.dirname(__file__), 'static'),
            'session_timeout':3600,

            'wx_token': 'wxly',
            'wx_test': True,
            }

import aiml
cur_dir = os.getcwd()
print 'cur_dir:', cur_dir
os.chdir('./res/alice')
alice = aiml.Kernel()
alice.learn("startup.xml")
alice.respond('LOAD ALICE')
os.chdir(cur_dir)
print 'cur_dir:', os.getcwd()

from handle import *

handlers=[
        (r'/wx', wx.WX),
        (r'/aiml', aiml.Alice),
        ]
