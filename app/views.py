from django.shortcuts import render, redirect, HttpResponse
import time
import pandas as pd 
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('rainfall_india.csv')
statewise = pd.read_csv('state1.csv')
months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
loc=[0,110,207,322,437,552,667,782,897,1012,1127,1142,1357,1472,1587,1702,1817,1932,2047,2162,2277,2392,2507,2622,2737,2852,2967,3082,3197,3312,3427,3542,3657,3772,3887,4002,4115]

regions=dataset.iloc[:,0].values
regions=list(regions)


def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 
regions=Remove(regions) 

region_frames=[]
maxcolumn=[]

for state in regions:    
	cond=dataset['SUBDIVISION']==state
	frame=dataset[cond]
	M=frame.set_index('YEAR')
	maxcolumn.append(M.max())
	region_frames.append(frame)


def createmodel(X_train,y_train,i):
    regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0,criterion='mse')
    regressor.fit(X_train, y_train[:,i])
    return regressor




def getmodel(region_id,i):
    start=loc[region_id]
    stop=loc[region_id+1]
    X = dataset.iloc[start:stop, 1].values
    y = dataset.iloc[start:stop, 2:19].values

    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(y[:, 0:])
    y[:, 0:] = imputer.transform(y[:, 0:])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
           
    
    X_train=X_train.reshape(-1,1)
    X_test =X_test.reshape(-1,1)
    obj=createmodel(X_train,y_train,i)
    return obj
    

# Create your views here.
def index(request):
	high_rain_region=statewise.loc[statewise['ANNUAL'].idxmax()][0]
	high_rain_region_value=int(statewise.loc[statewise['ANNUAL'].idxmax()][13])
	low_rain_region=statewise.loc[statewise['ANNUAL'].idxmin()][0]
	low_rain_region_value=int(statewise.loc[statewise['ANNUAL'].idxmin()][13])        
	stateindex=statewise.set_index('SUBDIVISION')
	stateindex =stateindex.drop(stateindex.columns[[12,13,14,15,16,17]], axis=1)
	p=stateindex.apply(lambda x: x.argmax(), axis=1)
	higest_rain_month=p[36]
	higest_rain_month_value = int(stateindex.iloc[36][p[36]])
	q=stateindex.apply(lambda x: x.argmin(), axis=1)
	lowest_rain_month=q[36]
	lowest_rain_month_value = int(stateindex.iloc[36][q[36]])
	yeardf=dataset.groupby('YEAR').sum()
	yeardf.reset_index(level=0, inplace=True)
	highest_rain_year=int(yeardf.loc[yeardf['ANNUAL'].idxmax()][0])
	highest_rain_year_value=int(yeardf.loc[yeardf['ANNUAL'].idxmax()]['ANNUAL'])
	lowest_rain_year=int(yeardf.loc[yeardf['ANNUAL'].idxmin()][0])
	lowest_rain_year_value=int(yeardf.loc[yeardf['ANNUAL'].idxmin()]['ANNUAL'])

	datalist=[high_rain_region,high_rain_region_value,low_rain_region,low_rain_region_value,higest_rain_month,lowest_rain_month,highest_rain_year,lowest_rain_year, highest_rain_year_value, lowest_rain_year_value,lowest_rain_month_value, higest_rain_month_value]
	yearframe=dataset.groupby('YEAR').mean()
	annualrain = list(yearframe.iloc[:,12].values)
	years=list(yearframe.index)
	monthlyrain=list(statewise.iloc[36,1:14].values)
	xvalues=years
	yvalues=annualrain
	y_var = [annualrain,monthlyrain]
	x_var = [years,months]

	return render(request, 'app/index.html', {'xvalues':x_var, 'yvalues':y_var,'data':datalist})

high_rain_year=[]
low_rain_year=[]
def highest_rain_year():
	i=0    
	while i<36:
		start=loc[i]
		stop=loc[i+1]
		high_rain_year.append(dataset.loc[dataset[start:stop]['ANNUAL'].idxmax()][1])
		i=i+1


def lowest_rain_year():
	i=0
	while i<36:
		start=loc[i]
		stop=loc[i+1]
		low_rain_year.append(dataset.loc[dataset[start:stop]['ANNUAL'].idxmin()][1])
		i=i+1


def region(request):
    region_id = int(request.GET['region'])
    month_id = int(request.GET['month'])
    year_id = int(request.GET['year'])
    region_name = regions[int(region_id)]
    xvalues=list(region_frames[region_id].iloc[:,1])
    yvalues=list(region_frames[region_id].iloc[:,month_id+1])
    y_var = yvalues
    x_var = json.dumps(xvalues)
    start=loc[region_id]
    stop=loc[region_id+1]
    X = dataset.iloc[start:stop, 1].values
    y = dataset.iloc[start:stop, 2:15].values
    obj=getmodel(region_id,month_id-1)
    d=y[:,month_id-1]
    y_pred=obj.predict(X.reshape(-1,1))
    
    acc=obj.score(X.reshape(-1,1),d.reshape(-1,1))
    acc=format(acc, "^.4f")
    x1_var = list(X)
    y2_var = list(y_pred)
    x2_var = list(X)
    y1_var = list(y[:,month_id-1])
    
    highest_rain_year()
    lowest_rain_year()
    highestyear=high_rain_year[region_id]
    lowestyear=low_rain_year[region_id]
    
    stateindex=statewise.set_index('SUBDIVISION')
    stateindex =stateindex.drop(stateindex.columns[[12,13,14,15,16,17]], axis=1)
    p=stateindex.apply(lambda x: x.argmax(), axis=1)
    higest_rain_month=p[region_id]
    q=stateindex.apply(lambda x: x.argmin(), axis=1)
    lowest_rain_month=q[region_id]

    
    input = {'high_month':higest_rain_month,'low_month':lowest_rain_month,'high_year':highestyear, 'low_year':lowestyear, 'name':region_name,'year_get':year_id ,'month_get':month_id, 'region_get':region_id, 'xvalues':x_var, 'yvalues':y_var,'month':months[month_id-1], 'xtrace1':x1_var, 'ytrace1':y1_var,'xtrace2':x2_var, 'ytrace2':y2_var,'accuracy':acc}
    return render(request, 'app/region.html', input)



def predict(request):
    num1=int(request.GET['year'])
    year=[[num1]]
    num2=int(request.GET['region'])
    region=num2
    region_name = regions[int(region)]
    
    start=loc[region]
    stop=loc[region+1]
    
    X = dataset.iloc[start:stop, 1].values
    y = dataset.iloc[start:stop, 2:19].values

    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(y[:, 0:])
    y[:, 0:] = imputer.transform(y[:, 0:])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
           
    
    X_train=X_train.reshape(-1,1)
    X_test =X_test.reshape(-1,1)
    y_pred=[]
    i=0
    while i<17:
        obj=createmodel(X_train,y_train,i)
        y_pred.append(int(obj.predict(year)))
        i=i+1
        
    yvalues=list(y_pred[:13])
    xvalues=list(months)
  #  xvalues=list(xvalues.append('ANNUAL'))
    piechart_val=list(y_pred[13:])
    piechart_label=['Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']
    pie=[piechart_val,piechart_label]
    y_var = yvalues
    x_var = json.dumps(xvalues)
    return render(request, 'app/predict.html', {'xvalues':x_var, 'yvalues':y_var, 'year':num1,'region':region_name, 'pie':pie})


def about(request):
	return render(request, 'app/about.html')
