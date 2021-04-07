# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:09:17 2021

@author: sowmi
"""

# import packages and read the input data

import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv("C:\\Users\\tussh\\Desktop\\Task\\Vidio.csv",delimiter=',')
data.isnull().sum()
data.dtypes
data.size
data.shape
data.columns
data.describe()
data.isnull().sum()
data.head()
data.dtypes

#drop column with more than 60% NAN values
data.drop(["city"],axis=1,inplace=True) 
data.drop(["hash_film_id"],axis=1,inplace=True)
data.drop(["utm_source"],axis=1,inplace=True)
data.drop(["utm_medium"],axis=1,inplace=True)
data.drop(["utm_campaign"],axis=1,inplace=True)
data.drop(["app_version"],axis=1,inplace=True)
data.drop(["stream_type"],axis=1,inplace=True)
data.drop(["film_title"],axis=1,inplace=True)
data.drop(["season_name"],axis=1,inplace=True)
data.drop(["genre_name"],axis=1,inplace=True)
data.drop(["Unnamed: 41"],axis=1,inplace=True)
data.drop(["Unnamed: 42"],axis=1,inplace=True)
data.drop(["Unnamed: 43"],axis=1,inplace=True)
data.drop(["Unnamed: 44"],axis=1,inplace=True)
data.drop(["Unnamed: 45"],axis=1,inplace=True)
data.shape
data.head()
data.isnull().sum()
data.isnull().mean().sort_values(ascending=False)



#Treating missing values in catergorical features
#Freaquent category imputation

def impute_nan(data,variable):
    most_frequent_category=data[variable].mode()[0]
    data[variable].fillna(most_frequent_category,inplace=True)

for feature in ['completed','player_name','os_name','browser_name','autoplay','category_name']:
    impute_nan(data,feature)
data.isnull().mean()

############################## treating alphanumeric ID ######################################################

####### hash_content_id #####
data.hash_content_id.head()
int_content_id = [hash(uid) for uid in data.hash_content_id]
print(int_content_id)
#data["hash_content_id"]= data["int_content_id"]
data['int_content_id'] = int_content_id
#len(data["int_content_id"].unique())#=10431 
#len(data["hash_content_id"].unique())#=10431 
data.drop(["hash_content_id"],axis=1,inplace=True)

################ hash_visit_id ###################
data.hash_visit_id.head()
int_visit_id = [hash(uid) for uid in data.hash_visit_id]
print(int_visit_id)
#data["hash_visit_id"]= data["int_visit_id"]
data['int_visit_id'] = int_visit_id
len(data["int_watcher_id"].unique())# = 106490
#len(data["hash_visit_id"].unique()) #= 106490
data.drop(["hash_visit_id"],axis=1,inplace=True)
#data.int_visit_id.isnull().sum()

##################### hash_play_id ############### 106823
int_play_id = [hash(uid) for uid in data.hash_play_id]
print(int_play_id)
data['int_play_id'] = int_play_id
data.drop(["hash_play_id"],axis=1,inplace=True)

########### hash_watcher_id #################### Unique 102996
int_watcher_id = [hash(uid) for uid in data.hash_watcher_id]
print(int_watcher_id)
data['int_watcher_id'] = int_watcher_id
data.drop(["hash_watcher_id"],axis=1,inplace=True)

################ hash_event_id ############### unique 106823
int_event_id = [hash(uid) for uid in data.hash_event_id]
print(int_event_id)
data['int_event_id'] = int_event_id
data.drop(["hash_event_id"],axis=1,inplace=True)
########################################################################


################ Treating timestamp values ############### play_time
data['play_Dates'] = pd.to_datetime(data['play_time']).dt.date
data['play_Time'] = pd.to_datetime(data['play_time']).dt.time
data.drop(["play_time"],axis=1,inplace=True)

################ Treating timestamp values ############### end_time
data['end_Dates'] = pd.to_datetime(data['end_time']).dt.date
data['end_Time'] = pd.to_datetime(data['end_time']).dt.time
data.drop(["end_time"],axis=1,inplace=True)


### Visualization####
    # comparison of completed shows
figure = plt.figure(figsize=(6,6))
sns.countplot(x="has_ad",data=data,palette='Set1')
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')
plt.title('Indicating the play has ad or not',size='13')
plt.show()

# comparison of completed shows
figure = plt.figure(figsize=(6,6))
sns.countplot(x="playback_location",data=data,palette='Set1')
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')
plt.title('where user play the video',size='13')
plt.show()

# comparison of completed shows
figure = plt.figure(figsize=(6,6))
sns.countplot(x="is_login",data=data,palette='Set1')
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')
plt.title('when user is logged_in when watch',size='13')
plt.show()
##########################################################
#Treating NAN in numeric values
#treating nan values in average_bitrate 

data["average_bitrate"].isnull().sum()
data['average_bitrate'].dropna().sample(data['average_bitrate'].isnull().sum(),random_state=0)
data[data['average_bitrate'].isnull()].index
def impute_nan(data,variable):
    data[variable+"_random"]=data[variable]
    ##It will have the random sample to fill the na
    random_sample=data[variable].dropna().sample(data[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=data[data[variable].isnull()].index
    data.loc[data[variable].isnull(),variable+'_random']=random_sample
impute_nan(data,"average_bitrate")
data.head()
data["average_bitrate_random"].isnull().sum()
#Visualisation
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111)
data['average_bitrate'].plot(kind='kde', ax=ax)
data.average_bitrate_random.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
data["average_bitrate"]=data["average_bitrate_random"]
data.drop(["average_bitrate_random"],axis=1,inplace=True)


#### Broswer__Version #############
data["browser_version"].isnull().sum()
data['browser_version'].dropna().sample(data['browser_version'].isnull().sum(),random_state=0)
data[data['browser_version'].isnull()].index
def impute_nan(data,variable):
    data[variable+"_random"]=data[variable]
    ##It will have the random sample to fill the na
    random_sample=data[variable].dropna().sample(data[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=data[data[variable].isnull()].index
    data.loc[data[variable].isnull(),variable+'_random']=random_sample
impute_nan(data,"browser_version")
data.head()
data["is_login"].unique()

#################################################################
############## Feature engineering  Label Encoding ##############
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
#​completed','player_name','os_name','browser_name','autoplay','category_name
data['is_login']= le.fit_transform(data['is_login']) 
data['has_ad']= le.fit_transform(data['has_ad']) 
data['is_premium']= le.fit_transform(data['is_premium'].astype(str)) 
data['content_type']= le.fit_transform(data['content_type'])
data['title']= le.fit_transform(data['title'])
data['referrer_group']= le.fit_transform(data['referrer_group'])
data['app_name']= le.fit_transform(data['app_name']) 
data['player_name']= le.fit_transform(data['player_name']) 
data['os_name']= le.fit_transform(data['os_name']) 
data['browser_name']= le.fit_transform(data['browser_name']) 
data['autoplay']= le.fit_transform(data['autoplay'].astype(str)) 
data['category_name']= le.fit_transform(data['category_name'])
data = pd.get_dummies(data,columns=['completed'],drop_first=True)
data['playback_location'].unique()
data["playback_location"] = data["playback_location"].astype('category')
data["playback_location"] = data["playback_location"].cat.codes
data['platform']=data['platform'].str.replace('-','').astype('category')
cleanup_nums = {"platform":     {"webmobile": 0, "webdesktop": 1, "appandroid": 2, "tvandroid": 3, "tvtizen": 4, "appios": 5, "tvwebos": 6}}
data = data.replace(cleanup_nums)                


final_outcome=data.to_csv("C:/Users/tussh/samplevidio.csv", index=False)

data = pd.read_csv("samplevidiolatest.csv")
#find labels in different features
for feature in data.columns[:]:
    print(feature,":",len(data[feature].unique()),'labels')
    
sns.pairplot(data= data,hue= login_is,size = 3)
    
#data["play_duration"]= data.play_duration.convert_objects(convert_numeric=True)
#data["play_duration"] = data.play_duration.astype(float)
#data.drop(["play_duration"],axis=1,inplace=True)  
  

################################################ SWEETVIZ ############################
pip install sweetviz
import sweetviz as sv
report = sv.analyze(data)
#display the report in html
report.show_html('report.html')


############### Model Building ##############################
X=data.drop(["is_login"],axis=1)
y=data['is_login']

################### Scaling the data  ######################

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)

#################### XGB Classifier ###############
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=10)

XGBC=xgb.XGBClassifier() 

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=XGBC.fit(X_train,y_train)

y_pred=model.predict(X_test)

np.mean(y_test==y_pred)*100
#91.84323025556215
print(classification_report(y_test,y_pred))
#acc = 92%
cohen_kappa_score(y_test,y_pred)
#0.779427823502241

## ROC Curve...
lr_roc_auc = roc_auc_score(y_test, XGBC.predict(X_test)) 
fpr, tpr, thresholds = roc_curve(y_test, XGBC.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='XGBoost(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .90%

####################### Logistic regression Algorithm....#################

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=logreg.fit(X_train,y_train)

y_pred=model.predict(X_test)
np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))
#acc= 89%
cohen_kappa_score(y_test,y_pred)
#  0.7200080214750806
################## RandomForest ####################
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =10)

rf= RandomForestClassifier(n_estimators=50, criterion='gini')

model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))
## Accuracy = 91%
accuracy_score(y_test, y_pred)
## Kappa Score
cohen_kappa_score(y_test,y_pred)
#0.7711980974323741


# Out of all the model the XGBoost performed the best and gave an acc of 92% 