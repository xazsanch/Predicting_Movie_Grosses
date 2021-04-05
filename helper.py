#Imports and Settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from operator import mul
import seaborn as sn

#Modeling
from sklearn.model_selection import KFold, RepeatedKFold,train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

#Pandas Settings to Display Rows and Cols
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 10) 

#Matplotlib Style Settings
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

#Pandas Standaridzer for Float Formatting
pd.options.display.float_format = '{:,.2f}'.format

# Data Pipeline
# Looping through data folder to concat CSV's into one big dataframe
# 13708 rows in dataset
# 12856 with Dom Gross
# Reinitialize HERE

for i,name in enumerate(glob.glob('data/*')):
    if i == 0:
        df = pd.read_csv(name)
    df2 = pd.read_csv(name)
    concat = pd.concat([df,df2],ignore_index=True)
    df = concat

#Drop Empty Dom Gross rows
df=df.dropna(subset=['Domestic   Gross'])
df=df.drop(['Desc'],axis=1)
df=df.drop(['Prod Co'],axis=1)

#Rename Columns
df.rename(columns={'Run-  time':'Runtime','Domestic   Gross':'Dom_Gross', 
                   'International   Gross':'Intl_Gross', 'Worldwide   Gross':'WW_Gross',
                   'Dom   Locs':'Dom_Locs','Opng Wknd   Gross': 'Opng_Wknd',
                  'Wknd   Days':'WkndDays','Wknd   Multi':'WkndMulti','Wk   Multi':'WkMulti'}, inplace=True)

#Dropping to prevent data leakage
df=df.drop(['WkndMulti'],axis=1)
df=df.drop(['WkMulti'],axis=1)
df=df.drop(['Intl_Gross'],axis=1)
df=df.drop(['WW_Gross'],axis=1)

#Set Rel Date to Date Time obj
df['Rel Date'] = pd.to_datetime(df['Rel Date'])
#Fixing Year Century error around 1970 (was applying to 2070)
df['Rel Date'] = df['Rel Date'].apply(lambda x: x.replace(year = x.year - 100) if x.year > 2036 else x)

#Separating Rel Date into Year and Month
df['Year'] = df['Rel Date'].dt.year
df['Month'] = df['Rel Date'].dt.month
df['Week Num'] = df['Rel Date'].dt.week

#4 Day opener binary field
df['4_Day_Open'] = df['WkndDays'].apply(lambda x: 1 if x==4 else 0)

#Rounding down for nearest Decade
from math import floor
def rounddown(x):
    return int(floor(x/10.0))*10

df['Decade'] = df.Year.apply(lambda x: rounddown(x))

#Applying 2D to NaN Format
df['Format'] = df['Format'].apply(lambda x: "2D" if pd.isnull(x) else x)

# Remove $ and convert to float
df[df.columns[6]] = df[df.columns[6]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[8]] = df[df.columns[8]].replace('[\$,]', '', regex=True).astype(float)

# Runtime Calculation function to separate H:MM into Minutes
# 367 NaN runtimes

def runtime_calc(x):
        runtime = x.split(':')
        runtime_int = [int(x1) for x1 in runtime]
        return runtime_int[0]*60 + runtime_int[1]

# Applying Runtime Function to Runtime Column
df['Runtime'] = df['Runtime'].apply(lambda x: runtime_calc(x) if type(x) == str else x)
df['Runtime_cat'] = df['Runtime'].apply(lambda x: 'U90Mins' if x < 90 else ('U2hrs' if x < 120 else 'O2Hrs'))

'''
# Creating Buckets for Runtime
df['U90Mins'] = df.Runtime.map(lambda x: 1.0 if x<90 else 0.0)
df['U2Hrs'] = df.Runtime.map(lambda x: 1.0 if x>=90 and x < 120 else 0.0)
df['O2Hrs'] = df.Runtime.map(lambda x: 1.0 if x>=120 else 0.0)
'''
#Splitting Genre, only focusing on Primary Genre
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])

# Removing Excess data of "Other Film Grosses"
df = df.drop(df[df.Picture.str.contains('Other') & df.Picture.str.contains('Film')].index)

#Removing Reissues
df = df.drop(df[df.Picture.str.contains('Reissue')].index)
df = df.drop(df[df.Picture.str.contains('Re ')].index)

# Converting CinemaScore metric from Letter Grade to Numeric
grades = {'A+':100,'A':95,'A-':90,
         'B+':87,'B':84,'B-':80,
         'C+':77,'C':74,'C-':70,
         'D+':67,'D':64,'D-':60,
         'F':50}

def convert_letter_grade(grade):
    return grades.get(grade,None)

# Applying convert_letter_grade function to Column
df['CS'] = df['CS'].apply(lambda x: convert_letter_grade(x))

#Inflation of Domestic Gross
# Updated with November 2019 CPI information
import cpi

# Not sure if we have to update every session, or if once is enough?
#cpi.update()

def inflate_gross(data,column):
    return data.apply(lambda x: cpi.inflate(x[column],x.Year),axis=1)

#New Column for Domestic Gross, accounting for inflation
df['inflated_dom_gross'] = inflate_gross(df,'Dom_Gross')
df['inflated_opng_wknd'] = inflate_gross(df,'Opng_Wknd')

#EDA Functions
#Plotting Avg Dom Gross by Col

def avgdombycol(col):
    fig,ax = plt.subplots(figsize=(10,10))
    plot_df = df.dropna(subset=[col])
    grouped = plot_df.groupby(col)['inflated_dom_gross'].mean().reset_index()
    grouped = grouped.sort_values(by=['inflated_dom_gross'],ascending=True)
    grouped = grouped.tail(10)
    ax.barh(grouped[col],grouped['inflated_dom_gross'],color='tab:blue')
    
    xlabels = ['${:,.0f}'.format(x) + 'M' for x in ax.get_xticks()/1000000]
    ax.set_xticklabels(xlabels)
    ax.set_title('Average Domestic Gross by {} (with Inflation)'.format(col))
    ax.set_xlabel('Domestic Gross')
    ax.set_ylabel(col)
    
    fig.tight_layout()

#Plot Histogram of Col
def hist_col(col,size, bins=5):
    fig, ax = plt.subplots(figsize=size)
    ax.hist(df[col],bins=bins)
    fig.tight_layout()

#Modelling Functions
def kfold_model(x,y,num,reg_model):
    n_splits=num
    kfold = KFold(n_splits)

    X_train,X_test, y_train, y_test = train_test_split(x,y,shuffle=False,stratify=None)

    rmse = []
    for train_index, test_index in kfold.split(X_train):
        reg = reg_model
        reg.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        y_pred = reg.predict(X_train.iloc[test_index])
        y_true = y_train.iloc[test_index]
        #mse = mean_squared_error(y_test,pred)
        rmse.append(mean_squared_error(y_true,y_pred,squared=False))

    #for err in rmse:
        #print('{:,.2f}'.format(err))
    print('Average RMSE with {} Kfolds: {:,.2f}'.format(n_splits,np.mean(rmse)))


    yhat = reg.predict(X_test)
    rmse_outter =mean_squared_error(y_test,yhat,squared=False)
    print('RMSE Outter: {:,.2f}'.format(np.mean(rmse_outter)))

    print('CV Score: {:,.2f}'.format(-1*np.mean(cross_val_score(reg,X_train,y_train,scoring='neg_root_mean_squared_error',cv=num))))

def lin_reg(x,y,num=5,reg_model):
    n_splits=num

    X_train,X_test, y_train, y_test = train_test_split(x,y,shuffle=False,stratify=None)

    reg = reg_model
    reg.fit(X_train,y_train)

    yhat = reg.predict(X_test)
    rmse_outter =mean_squared_error(y_test,yhat,squared=False)
    print('RMSE Outter: {:,.2f}'.format(np.mean(rmse_outter)))

    print('CV Score: {:,.2f}'.format(-1*np.mean(cross_val_score(reg,X_train,y_train,scoring='neg_root_mean_squared_error',cv=num))))

#Apply RobustScaler to Columns
def robust_transform(x):
    new_x = x.copy()
    robust = RobustScaler()
    continuous = x[['Dom_Locs','Runtime','score','inflated_opng_wknd']]
    transformed_data = robust.fit_transform(continuous)
    
    new_x['Dom_Locs'] = transformed_data[:,0].reshape(-1,1)
    new_x['Runtime'] = transformed_data[:,1].reshape(-1,1)
    new_x['score'] = transformed_data[:,2].reshape(-1,1)
    new_x['inflated_opng_wknd'] = transformed_data[:,3].reshape(-1,1)
    
    return new_x

#Perform Lasso Regression and Return Score
def lasso_score(x,y,num_it):
    #Split Data
    X_train,X_test, y_train, y_test = train_test_split(x,y,shuffle=False,stratify=None)
    lasso = Lasso(max_iter=num_it)
    lasso.fit(X_train, y_train)
    
    #Model Score
    model_score = lasso.score(X_test,y_test)
    print('Model Score: {:,.4f}'.format(model_score))
    
    #Preduction with Model
    yhat = lasso.predict(X_test)
    lasso_rmse =mean_squared_error(y_test,yhat,squared=False)
    print('Lasso RMSE: {:,.2f}'.format(np.mean(lasso_rmse)))
    return [num_it, model_score,lasso_rmse]

#Return Coeffecient DF
def coeff_df(model,x,num):
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    coeff_df = coeff_df.sort_values(by='Coefficient',ascending=False).head(num)
    return coeff_df

#GridSearch, return Feature Importances
def gsearch(model,param_grid,X_unseen,y_unseen,X_train,y_train):    
    clf = GridSearchCV(model,param_grid,verbose=1,n_jobs=-1,scoring ='neg_root_mean_squared_error')

    clf.fit(X_train,y_train)
    print(clf.best_params_)

    #Compare GridSearch on test data and train data
    print('{} GridSearch RMSE: {:,.2f}'.format(model,-1*clf.best_score_))
    print('Train RMSE: {:,.2f}'.format(-1* clf.score(X_train,y_train)))
    print('Test RMSE: {:,.2f}'.format(-1* clf.score(X_test,y_test)))
    
    print('Unseen Test RMSE: {:,.2f}'.format(-1*clf.score(X_unseen,y_unseen)))
    return clf.best_estimator_.feature_importances_

#Return Best Features
def best_features(df,fi,greater_val,top=10):
    best_feat = df.columns[fi> greater_val]
    best_feat_vals=fi[fi > greater_val]
    best_feat_df = pd.Series(best_feat_vals,best_feat).head(top)
    return best_feat_df.sort_values(ascending=True,inplace=True)

def plot_best_features(df,modeltype,greater_val,axis=(0,0.2,6)):
    fig,ax = plt.subplots(figsize=(10,10))
    label =np.linspace(axis)
    ax.barh(df.index,df)
    ax.set_xticks(label)
    ax.set_xlabel('Fraction of Samples Affected')
    ax.set_title('{} Feature Importance Greater than {}'.format(modeltype,greater_val))
    fig.tight_layout()
