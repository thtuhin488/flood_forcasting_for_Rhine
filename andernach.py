import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

excel_file = 'sheet 1.xlsx'
Dt = pd.read_excel(excel_file,usecols=['Date'])
E = pd.read_excel(excel_file,usecols=['Evaporation'])
P = pd.read_excel(excel_file,usecols=['Precipitation'])
D = pd.read_excel(excel_file,usecols=['Discharge'])
T = pd.read_excel(excel_file,usecols=['Temperature'])
Dt16 = Dt.loc[0:1460]
E16 = E.loc[0:1460]
P16 = P.loc[0:1460]
D16 = D.loc[0:1460]
T16 = T.loc[0:1460]
fig,ax1 =plt.subplots()

ax1.set_xlabel('Date')
ax1.set_ylabel('Discharge(m3/s)')
ax1.plot(Dt16,D16,linewidth=.6,color='red')
ax2=ax1.twinx()
#ax2.set_ylabel('Precipitation(mm)')
#ax2.plot(Dt16,P16,linewidth=.8)
#ax2.set_ylabel('Evaporation(mm)')
#ax2.plot(Dt16,E16,linewidth=.7)
ax2.set_ylabel('Temperature(°C)')
ax2.plot(Dt16,T16,linewidth=.7)


#df = pd.read_excel(excel_file,usecols=['Date','Evaporation','Precipitation','Discharge','Temperature'])
#df1=df.loc[0:1460]
#df1 = df.to_json()
#print(df.dtypes)
#print(df.at[5,'Discharge'])
#print (df)
#df1=df.loc[0:1460]
#p = df1.plot.line(x='Date', y='Evaporation')
#q = df1.plot.line(x='Date', y='Precipitation')
#plt.plot(p)
#plt.show() ))
#ax = plt.subplots()
#df1.plot(x='Date', y='Temperature')


#plt.plot(Dt16,D16,'b')
#plt.plot(Dt16,D16,'b')
#plt.plot(Dt16,E16,'g' )
#plt.plot(Dt16,P16,'black')
#plt.plot(Dt16,T16,'orange')


plt.show()

#print ()


#
