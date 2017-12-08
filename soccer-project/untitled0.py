# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:35:45 2017

@author: ikats
"""

import sqlite3
import pandas as pd
import sys
import os
import datetime as dt
import numpy as np
import matplotlib.dates as mdates #import DateFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math

#Sqlite database connection
#sqliteFile = "/home/data602/Desktop/Pavan/Data602/Final/data602-final/app/database/database.sqlite"
sqliteFile = "C:\\Temp\\CUNY\\data602-final\\app\\database\\database.sqlite"

conn = sqlite3.connect(sqliteFile)
query = ("SELECT * FROM `Player_Attributes` WHERE player_api_id = 30981")
try:
   playerDf = pd.read_sql_query(query, conn)
   conn.commit()
   conn.close()
except sqlite3.Error as er:
   conn.rollback()
   conn.close()

del conn

#Function converts date to number format
def bytespdate2num(fmt, encoding='utf-8'):
   strconverter = mdates.strpdate2num(fmt)
   def bytesconverter(b):
       s = b.decode(encoding)
       return strconverter(s)
   return bytesconverter

#Convert text to date
playerDf['Date_pd'] = pd.to_datetime(playerDf.date)
playerDf = playerDf.sort_values(by='Date_pd', ascending=True)

#Convert date to string format
playerDf['date1'] = playerDf['Date_pd'] .apply(lambda x: dt.datetime.strftime(x, '%m/%d/%Y'))
   
playerDf['data'] = playerDf.date1.str.cat(playerDf.finishing.astype("str"), sep=',')
resultList = playerDf.data.tolist()

x, y = np.loadtxt(resultList,
                 delimiter=',',
                 unpack=True,
                 converters={0: bytespdate2num('%m/%d/%Y')})

plt.close('all')
fig, ax = plt.subplots()
ax.grid(True)
ax.plot(x, y)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.xlabel('Date')
plt.ylabel('Finishing Skill')
plt.title('Change in Finishing Skill')
fig.autofmt_xdate()
plt.show()