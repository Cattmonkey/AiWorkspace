# encoding=utf-8
from urllib import urlopen
#import tushare as ts  
import pandas as pd  
from sqlalchemy import create_engine
import pymysql
from pymysql import IntegrityError
import settings
import pandas as pd

class Data_Mysql () :
    
    def __init__ (self):
        #self.engine = create_engine('mysql://root:pcm@192.168.100.244?charset=utf8')#用sqlalchemy创建引擎  
        #df.to_sql('tick_data',engine,if_exists='append')#存入数据库，这句有时候运行一次报错，运行第二次就不报错了，不知道为什么  
        #df1 = pd.read_sql('tick_data',engine)#从数据库中读取表存为DataFrame  
        self.host = settings.MYSQL_HOST
        self.user = settings.MYSQL_USER
        self.password = settings.MYSQL_PASSWORD
        self.db = settings.MYSQL_DB        
        self.conn = None
        p = pymysql()

        
    def read (self, table, column, LIMIT = ' '):
        if not self.conn:
            self.connect ()
        sql = "select " + str (column) + " from " + str (table) + ' ' + LIMIT
        df = pd.read_sql (sql, self.conn)
        self.close ()
        return df
    
    def connect (self):
        self.conn = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.db, charset="utf8")
        pass
    
    def close(self):
        self.conn.close()    
        
if __name__ == "__main__":
    
    readMysql = Data_Mysql ()
    print (readMysql.read ("Xcar_Content", "content"))