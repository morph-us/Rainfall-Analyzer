import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv('rainfall_india.csv')
#print(df)

xlist=list(df['SUBDIVISION'].unique())

trace1 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['JAN'],name='JAN',marker={'color':'#cb4335'})
trace2 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['FEB'],name='FEB',marker={'color':'#884ea0'})
trace3 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['MAR'],name='MAR',marker={'color':'#2980b9'})
trace4 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['APR'],name='APR',marker={'color':'#3498db'})
trace5 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['MAY'],name='MAY',marker={'color':'#76d7c4'})
trace6 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['JUN'],name='JUN',marker={'color':'#1abc9c'})
trace7 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['JUL'],name='JUL',marker={'color':'#f1c40f'})
trace8 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['AUG'],name='AUG',marker={'color':'#d35400'})
trace9 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['SEP'],name='SEP',marker={'color':'#f39c12'})
trace10 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['OCT'],name='OCT',marker={'color':'#17202a'})
trace11 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['NOV'],name='NOV',marker={'color':'#bdc3c7'})
trace12 = go.Bar(x=xlist,y=df.groupby('SUBDIVISION').mean()['DEC'],name='DEC',marker={'color':'#34495e'})
data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]
layout = go.Layout(barmode='stack', margin=dict(l=20, r=20, t=10, b=100), height=400)
fig = go.Figure(data=data,layout=layout)
pyo.plot(fig,filename='Rainfall.html')
