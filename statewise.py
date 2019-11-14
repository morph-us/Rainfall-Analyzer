#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:05:14 2019

@author: harshal
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import pickle


regressor=[[]]
mean_train_res=[[]]
mean_test_res=[[]]

dataset = pd.read_csv('rainfall in india 1901-2015.csv')
month=['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec','Annual']

def createmodel(X_train,y_train,i):
    regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0,criterion='mae')
    regressor.fit(X_train, y_train[:,i])
    return regressor

def saveimg(X_train,y_test,X_test,i,regressor,statename):
    X_grid = np.arange(min(X_train), max(X_train), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    y_pred=regressor.predict(X_grid)
    plt.scatter(X_test, y_test[:,i] ,color = 'red')
    plt.plot(X_grid, y_pred, color = 'blue')
    plt.title('Years vs '+ month[i]+' rainfall '+statename)
    plt.xlabel('Years')
    plt.ylabel(' Rainfall(mm)')
    #plt.show()
    #plt.savefig(statename+month[i]+' rainfall.png',dpi=300)
    plt.close()



def state(start,stop,j,statename):
    
    X = dataset.iloc[start:stop, 1].values
    y = dataset.iloc[start:stop, 2:15].values
    
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(y[:, 0:])
    y[:, 0:] = imputer.transform(y[:, 0:])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
           
    
    X_train=X_train.reshape(-1,1)
    X_test =X_test.reshape(-1,1)
     
    

    regressor.append([])
    mean_test_res.append([])    
    mean_train_res.append([])    
    i=0
    while i<13:
        obj=createmodel(X_train,y_train,i)
        regressor[j].append(obj)
        saveimg(X_train,y_test,X_test,i,obj,statename)
        
        
            
              
        predicted_values=obj.predict(X_test)
        residuals = predicted_values-y_test[:,i]
        mean_test_res[j].append(np.mean(np.abs(residuals)))
        print('MAD (TEST Data): '+month[i]+' = ' + str(mean_test_res[j][i]))
        
         
        predicted_values=obj.predict(X_train)
        residuals = predicted_values-y_train[:,i]
        mean_train_res[j].append(np.mean(np.abs(residuals)))
        print('MAD (Train Data): '+month[i]+' = ' + str(mean_train_res[j][i]))
        i=i+1

        
regions=dataset.iloc[:,0].values
regions=list(regions)
 
def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 
 
    

regions=Remove(regions) 

loc=[0,110,207,332,437,552,667,782,897,1012,1127,1142,1357,1472,1587,1702,1817,1932,2047,2162,2277,2392,2507,2622,2737,2852,2967,3082,3197,3312,3427,3542,3657,3772,3887,4002,4115]
i=0
#loc=[0,110,207,332,437]
while i<3:  
    state(loc[i],loc[i+1],i,regions[i])
    i=i+1
    
    
    
year=[[2019]]
region=regions[0]
period=month[0]

    
#state,month,year=input("Enter state , month , year").split()
#x=[[int(year)]]
#y_pred=regressor[int(state)][int(month)].predict(x)
y_pred=[[]]
j=0
while j<3:
    y_pred.append([])
    i=0
    while i<13:
        y_pred[j].append((float(regressor[j][i].predict(year))))
        i=i+1
    j=j+1    
        

df=pd.DataFrame(y_pred,columns=list(dataset.columns)[2:15])



filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#regressor = pickle.load(open(filename, 'rb'))
