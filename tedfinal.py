# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:08:16 2023

@author: danush
"""

import os
import pandas as pd
import random
random.seed(1)
#read in the dataset 

raw_dataset = pd.read_csv("2015.csv")
raw_dataset.shape

#here we check that the data loaded in is in the correct format
pd.set_option('display.max_columns', 500)
raw_dataset.head()
raw_numeric = raw_dataset.select_dtypes(include=['number'])
print(raw_numeric)

missing_values = raw_numeric.isnull().sum()
percentage_missing = (missing_values / raw_numeric.shape[0]) * 100
columns_to_drop = percentage_missing[percentage_missing > 70].index.tolist()
#Dropping the columns in the raw_numeric dataset
raw_numeric = raw_numeric.drop(columns_to_drop, axis=1)
print(raw_numeric)

raw_numeric = raw_numeric.reindex(columns=['_MICHD', '_RFHYPE5',  'TOLDHI2', 
                                           '_CHOLCHK', '_BMI5', 'SMOKE100', 
                                           'CVDSTRK3', 'DIABETE3', '_TOTINDA', 
                                           '_FRTLT1', '_VEGLT1','_RFDRHV5', 
                                           'HLTHPLN1', 'MEDCOST', 'GENHLTH', 
                                           'MENTHLTH', 'PHYSHLTH', 'DIFFWALK',
                                           'SEX', '_AGEG5YR', 'INCOME2','_STATE' ])
raw_numeric.shape
raw_numeric.columns
#here we are dropping missing values

raw_numeric.shape
raw_numeric = raw_numeric.dropna()
raw_numeric.shape
#DATA STANDARDIZATION

# _MICHD
#Change 2 to 0 because this means did not have MI or CHD
raw_numeric['_MICHD'] = raw_numeric['_MICHD'].replace({2: 0})
raw_numeric._MICHD.unique()
#1 _RFHYPE5
#Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
raw_numeric['_RFHYPE5'] = raw_numeric['_RFHYPE5'].replace({1:0, 2:1})
raw_numeric = raw_numeric[raw_numeric._RFHYPE5 != 9]
raw_numeric._RFHYPE5.unique()

#2 TOLDHI2
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
raw_numeric['TOLDHI2'] = raw_numeric['TOLDHI2'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric.TOLDHI2 != 7]
raw_numeric = raw_numeric[raw_numeric.TOLDHI2 != 9]
raw_numeric.TOLDHI2.unique()



#4 _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
raw_numeric['_BMI5'] = raw_numeric['_BMI5'].div(100).round(0)
raw_numeric._BMI5.unique()

#5 SMOKE100
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
raw_numeric['SMOKE100'] = raw_numeric['SMOKE100'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric.SMOKE100 != 7]
raw_numeric = raw_numeric[raw_numeric.SMOKE100 != 9]
raw_numeric.SMOKE100.unique()

#6 CVDSTRK3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
raw_numeric['CVDSTRK3'] = raw_numeric['CVDSTRK3'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric.CVDSTRK3 != 7]
raw_numeric = raw_numeric[raw_numeric.CVDSTRK3 != 9]
raw_numeric.CVDSTRK3.unique()

#7 DIABETE3
# going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
# Remove all 7 (dont knows)
# Remove all 9 (refused)
raw_numeric['DIABETE3'] = raw_numeric['DIABETE3'].replace({2:0, 3:0, 1:2, 4:1})
raw_numeric = raw_numeric[raw_numeric.DIABETE3 != 7]
raw_numeric = raw_numeric[raw_numeric.DIABETE3 != 9]
raw_numeric.DIABETE3.unique()

#8 _TOTINDA
# 1 for physical activity
# change 2 to 0 for no physical activity
# Remove all 9 (don't know/refused)
raw_numeric['_TOTINDA'] = raw_numeric['_TOTINDA'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric._TOTINDA != 9]
raw_numeric._TOTINDA.unique()

#9 _FRTLT1
# Change 2 to 0. this means no fruit consumed per day. 1 will mean consumed 1 or more pieces of fruit per day 
# remove all dont knows and missing 9
raw_numeric['_FRTLT1'] = raw_numeric['_FRTLT1'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric._FRTLT1 != 9]
raw_numeric._FRTLT1.unique()

#10 _VEGLT1
# Change 2 to 0. this means no vegetables consumed per day. 1 will mean consumed 1 or more pieces of vegetable per day 
# remove all dont knows and missing 9
raw_numeric['_VEGLT1'] = raw_numeric['_VEGLT1'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric._VEGLT1 != 9]
raw_numeric._VEGLT1.unique()

#11 _RFDRHV5
# Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
# remove all dont knows and missing 9
raw_numeric['_RFDRHV5'] = raw_numeric['_RFDRHV5'].replace({1:0, 2:1})
raw_numeric = raw_numeric[raw_numeric._RFDRHV5 != 9]
raw_numeric._RFDRHV5.unique()


#12 HLTHPLN1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 7 and 9 for don't know or refused
raw_numeric['HLTHPLN1'] = raw_numeric['HLTHPLN1'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric.HLTHPLN1 != 7]
raw_numeric = raw_numeric[raw_numeric.HLTHPLN1 != 9]
raw_numeric.HLTHPLN1.unique()



#14 GENHLTH
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
raw_numeric = raw_numeric[raw_numeric.GENHLTH != 7]
raw_numeric = raw_numeric[raw_numeric.GENHLTH != 9]
raw_numeric.GENHLTH.unique()

#15 MENTHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
raw_numeric['MENTHLTH'] = raw_numeric['MENTHLTH'].replace({88:0})
raw_numeric = raw_numeric[raw_numeric.MENTHLTH != 77]
raw_numeric = raw_numeric[raw_numeric.MENTHLTH != 99]
raw_numeric.MENTHLTH.unique()

#16 PHYSHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
raw_numeric['PHYSHLTH'] = raw_numeric['PHYSHLTH'].replace({88:0})
raw_numeric = raw_numeric[raw_numeric.PHYSHLTH != 77]
raw_numeric = raw_numeric[raw_numeric.PHYSHLTH != 99]
raw_numeric.PHYSHLTH.unique()

#17 DIFFWALK
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
raw_numeric['DIFFWALK'] = raw_numeric['DIFFWALK'].replace({2:0})
raw_numeric = raw_numeric[raw_numeric.DIFFWALK != 7]
raw_numeric = raw_numeric[raw_numeric.DIFFWALK != 9]
raw_numeric.DIFFWALK.unique()

#18 SEX
# in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
# change 2 to 0 (female as 0). Male is 1
raw_numeric['SEX'] = raw_numeric['SEX'].replace({2:0})
raw_numeric.SEX.unique()

#19 _AGEG5YR
# already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
# remove 14 because it is don't know or missing
raw_numeric = raw_numeric[raw_numeric._AGEG5YR != 14]
raw_numeric._AGEG5YR.unique()



#21 INCOME2
# Variable is already ordinal with 1 being less than $10,000 all the way up to 8 being $75,000 or more
# Remove 77 and 99 for don't know and refused
raw_numeric = raw_numeric[raw_numeric.INCOME2 != 77]
raw_numeric = raw_numeric[raw_numeric.INCOME2 != 99]
raw_numeric.INCOME2.unique()

raw_numeric.shape

raw_numeric.groupby(['_MICHD']).size()

#Rename the columns to make them more readable
final =raw_numeric.rename(columns = {'_MICHD':'Heart_Disease', 
                                         '_RFHYPE5':'HighBP',  
                                         'TOLDHI2':'High_Cholestrol', 
                                         '_BMI5':'BMI', 
                                         'SMOKE100':'Smoker', 
                                         'CVDSTRK3':'Stroke', 
                                         'DIABETE3':'Diabetes', 
                                         '_TOTINDA':'Physical_Activity', 
                                         '_FRTLT1':'Fruits',
                                         '_VEGLT1':"Veggies", 
                                         '_RFDRHV5':'Alcohol_Consumption', 
                                         'HLTHPLN1':'Healthcare', 
                                         'GENHLTH':'General_Health', 
                                         'MENTHLTH':'Mental_Health', 
                                         'PHYSHLTH':'Physical_Health',
                                         'DIFFWALK':'Difficulty_Walk', 
                                         'SEX':'Sex', '_AGEG5YR':'Age', 
                                         'INCOME2':'Income','_STATE':'State' })
# How many people have a heart disease in the data
final.groupby(['Heart_Disease']).size() # 10.4%
final.to_csv('heart_disease_health_indicators2014.csv', sep=",", index=False)
final.columns

# Logistic Regression on the final dataset
# Finding differentt x variables for logistic regression
# Trial 1 : 
final=pd.read_csv('heart_disease_health_indicators2014.csv')
x=final[['HighBP', 'High_Cholestrol', '_CHOLCHK', 'BMI',
       'Smoker', 'Stroke', 'Diabetes', 'Physical_Activity', 'Fruits',
       'Veggies', 'Alcohol_Consumption', 'Healthcare', 'MEDCOST',
       'General_Health', 'Mental_Health', 'Physical_Health', 'Difficulty_Walk',
       'Sex', 'Age', 'State', 'Income']]
y = final['Heart_Disease']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=43)
from sklearn.linear_model import LogisticRegression

#------- Logistic Regression Model------------
logmodel = LogisticRegression(solver='liblinear')
#grid_search = GridSearchCV(logmodel, param_grid, scoring='f1', cv=5) ## there are diff solvers
#grid_search.fit(x_train, y_train)
#best_params = grid_search.best_params_
#bestlogmodel=LogisticRegression(**best_params)
logmodel.fit(x_train,y_train)
probabilities = logmodel.predict_proba(x_test)
y_pred = logmodel.predict(x_test)

from sklearn.metrics import (accuracy_score,precision_score,recall_score,
                             confusion_matrix,f1_score)

f1_score(y_test,y_pred) # Logistic regression f1 score is 0.21
accuracy_score(y_test,y_pred) # Accuracy is 90.62%

recall_score(y_test,y_pred) # Recall is 12.35%

precision_score(y_test,y_pred) # Precision is 53.82%
c_mat=pd.DataFrame(confusion_matrix(y_test,y_pred),index=['Actual:0','Actual:1'],
             columns=['Pred:0','Pred:1'])
c_mat
#         Pred:0  Pred:1
#Actual:0   68861     684
#Actual:1    6452     877

#new model confusion matrix
#          Pred:0  Pred:1
#Actual:0   51770   17877
#Actual:1    1568    5659

#------- Decision Tree & Random Forests -------------------

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(x_train,y_train)

y_pred_train = dt.predict(x_train)

from sklearn.metrics import f1_score
f1_score(y_train,y_pred_train)

y_pred_test = dt.predict(x_test)

from sklearn.metrics import f1_score
f1_score(y_test,y_pred_test)
dt.tree_.max_depth


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1,n_estimators=200)
rfc.fit(x_train,y_train)
y_pred_test = rfc.predict(x_test)
f1_score(y_test,y_pred_test)

accuracy_score(y_test,y_pred) # Accuracy is 90.62%

recall_score(y_test,y_pred) # Recall is 11.72%

precision_score(y_test,y_pred)

final.columns
#VISUALIZATIONS
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x='Heart_Disease', y='Mental_Health', hue='Smoker', data=final)
plt.xlabel('heart')
plt.ylabel('mental')
plt.title('title')
plt.show()

sns.scatterplot(x='Heart_Disease', y='BMI', hue='Smoker', data=final)
plt.xlabel('heart')
plt.ylabel('bmi')
plt.title('Title')
plt.show()

sns.scatterplot(x='Heart_Disease', y='Physical_Health', hue='Smoker', data=final)
plt.xlabel('heart')
plt.ylabel('physical health')
plt.title('Title')
plt.show()

sns.barplot(x='Sex', y='Heart_Disease', hue='Smoker', data=final)
plt.xlabel('gender')
plt.ylabel('heart')
plt.title('plot')
plt.legend(title='plot')
plt.show()

#cleaning state for tableau visualizations
state_dict = {
    1: 'Alabama',
    2: 'Alaska',
    4: 'Arizona',
    5: 'Arkansas',
    6: 'California',
    8: 'Colorado',
    9: 'Connecticut',
    10: 'Delaware',
    11: 'District of Columbia',
    12: 'Florida',
    13: 'Georgia',
    15: 'Hawaii',
    16: 'Idaho',
    17: 'Illinois',
    18: 'Indiana',
    19: 'Iowa',
    20: 'Kansas',
    21: 'Kentucky',
    22: 'Louisiana',
    23: 'Maine',
    24: 'Maryland',
    25: 'Massachusetts',
    26: 'Michigan',
    27: 'Minnesota',
    28: 'Mississippi',
    29: 'Missouri',
    30: 'Montana',
    31: 'Nebraska',
    32: 'Nevada',
    33: 'New Hampshire',
    34: 'New Jersey',
    35: 'New Mexico',
    36: 'New York',
    37: 'North Carolina',
    38: 'North Dakota',
    39: 'Ohio',
    40: 'Oklahoma',
    41: 'Oregon',
    42: 'Pennsylvania',
    44: 'Rhode Island',
    45: 'South Carolina',
    46: 'South Dakota',
    47: 'Tennessee',
    48: 'Texas',
    49: 'Utah',
    50: 'Vermont',
    51: 'Virginia',
    53: 'Washington',
    54: 'West Virginia',
    55: 'Wisconsin',
    56: 'Wyoming',
    66: 'Guam',
    72: 'Puerto Rico'
}

# Load the CSV file into a pandas DataFrame
df3 = pd.read_csv('heart_disease_health_indicators2014.csv')

# Replace state codes with state names
df['State'] = df['State'].map(state_dict)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_heart_disease.csv', index=False)



# Load the csv file
df4 = pd.read_csv('updated_heart_disease.csv')

# Define a dictionary of state capitals and their corresponding coordinates
coordinates = {
    'Alabama': (32.3792, -86.3077),
    'Alaska': (58.3019, -134.4197),
    'Arizona': (33.4484, -112.0740),
    'Arkansas': (34.7465, -92.2896),
    'California': (38.5816, -121.4944),
    'Colorado': (39.7392, -104.9903),
    'Connecticut': (41.7637, -72.6851),
    'Delaware': (39.1582, -75.5244),
    'District of Columbia': (38.9072, -77.0369),
    'Florida': (30.4383, -84.2807),
    'Georgia': (33.7490, -84.3880),
    'Hawaii': (21.3069, -157.8583),
    'Idaho': (43.6187, -116.2146),
    'Illinois': (39.7817, -89.6501),
    'Indiana': (39.7684, -86.1581),
    'Iowa': (41.6005, -93.6091),
    'Kansas': (39.0473, -95.6752),
    'Kentucky': (38.2009, -84.8733),
    'Louisiana': (30.4515, -91.1871),
    'Maine': (44.3106, -69.7795),
    'Maryland': (38.9784, -76.4922),
    'Massachusetts': (42.3601, -71.0589),
    'Michigan': (42.7325, -84.5555),
    'Minnesota': (44.9537, -93.0892),
    'Mississippi': (32.2988, -90.1848),
    'Missouri': (38.5767, -92.1735),
    'Montana': (46.5927, -112.0361),
    'Nebraska': (40.8136, -96.7026),
    'Nevada': (39.1638, -119.7674),
    'New Hampshire': (43.2081, -71.5376),
    'New Jersey': (40.2206, -74.7597),
    'New Mexico': (35.6870, -105.9378),
    'New York': (42.6526, -73.7562),
    'North Carolina': (35.7796, -78.6382),
    'North Dakota': (46.8083, -100.7837),
    'Ohio': (39.9612, -82.9988),
    'Oklahoma': (35.4676, -97.5164),
    'Oregon': (44.9429, -123.0351),
    'Pennsylvania': (40.2732, -76.8867),
    'Rhode Island': (41.8240, -71.4128),
    'South Carolina': (34.0007, -81.0348),
    'South Dakota': (44.3683, -100.3510),
    'Tennessee': (36.1627, -86.7816),
    'Texas': (30.2672, -97.7431),
    'Utah': (40.7608, -111.8910),
    'Vermont': (44.2601, -72.5754),
    'Virginia': (37.5407, -77.4360),
    'Washington': (47.0379, -122.9007),
    'West Virginia': (38.3498, -81.6326),
    'Wisconsin': (43.0731, -89.4012),
    'Wyoming': (41.1400, -104.8203),
    'Guam': (13.4757, 144.7497),
    'Puerto Rico': (18.4655, -66.1057)
    }

df4['Latitude'] = df4['State'].map(lambda x: coordinates.get(x, (None, None))[0])
df4['Longitude'] = df4['State'].map(lambda x: coordinates.get(x, (None, None))[1])

df4.to_csv('updated_updated_heart_disease.csv')

