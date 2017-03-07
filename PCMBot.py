# -*- coding: utf-8 -*-
import aiml
import os

os.chdir('./alice')

#os.chdir('./alice') #切换工作目录到alice文件夹下，视具体情况而定
#alice = aiml.Kernel()
#alice.learn("startup.xml")
#alice.respond('LOAD ALICE')
#alice.respond('hello')
# 创建Kernel()和 AIML 学习文件

#pcmKernel = aiml.Kernel()
#pcmKernel.learn("startup.xml")
#pcmKernel.respond("LOAD ALICE")
pcmKernel = aiml.Kernel()
if os.path.isfile("bot_brain.brn"):
    pcmKernel.bootstrap(brainFile = "bot_brain.brn")
else:
    pcmKernel.bootstrap(learnFiles = "startup.xml", commands = "LOAD ALICE")
    pcmKernel.saveBrain("bot_brain.brn")
    
sessionId = 12345

#sessionId = 12345
 
## 将会话信息作为字典
## 包含输入输出的历史像已知谓词那样
#sessionData = kernel.getSessionData(sessionId)
 
## 每个会话ID需要一个唯一的值
## 用会话中机器人已知的人或事给谓词命名
## 机器人已经知道你叫"Billy"而你的狗叫"Brandy"
#kernel.setPredicate("dog", "Brandy", sessionId)
#clients_dogs_name = kernel.getPredicate("dog", sessionId)
 
#kernel.setBotPredicate("hometown", "127.0.0.1")
#bot_hometown = kernel.getBotPredicate("hometown")

while True:
    message = raw_input("Enter your message to the bot: ")
    if message == "quit":
        exit()
    elif message == "save":
        pcmKernel.saveBrain("bot_brain.brn")
    else:
        bot_response = pcmKernel.respond(message, sessionId)
        print bot_response