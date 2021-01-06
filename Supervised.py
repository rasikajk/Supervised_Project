#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV    
from collections import Counter 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir('D:\Trim 5')
os.getcwd()


# In[3]:


test = pd.read_csv('test.csv')


# In[4]:


test.head(5)


# In[5]:


train  = pd.read_csv("train.csv",na_values={"pickup_datetime":"43"})


# In[6]:


train.head(5)


# In[7]:


test.shape,train.shape


# In[8]:


train.info()


# In[9]:


train.describe()


# In[10]:


test.info()


# In[11]:


test.describe()


# In[12]:


train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce")


# In[13]:


train.info()


# In[15]:


train.dropna(subset= ["pickup_datetime"])   


# In[16]:


train['fare_amount'].sort_values(ascending=False)


# In[17]:


sum(train['fare_amount']==0)


# In[18]:


sum(train['fare_amount']>500)


# In[19]:


train=train.drop(train[train['fare_amount']<1].index,axis=0)


# In[20]:


sum(train['fare_amount']==0)


# In[21]:


train=train.drop(train[train['fare_amount']>500].index,axis=0)


# In[22]:


sum(train['fare_amount']>500)


# In[23]:


train[train['fare_amount'].isnull()]


# In[24]:


sum(train['fare_amount'].isnull())


# In[25]:


plt.figure(figsize=(15,7))
sns.countplot("passenger_count", data=train)


# In[26]:


train["passenger_count"].sort_values(ascending= True)


# In[27]:


sum(train['passenger_count']<1)


# In[28]:


sum(train['passenger_count']>6)


# In[29]:


train = train.drop(train[train['passenger_count']<1].index, axis=0)


# In[30]:


train = train.drop(train[train['passenger_count']>6].index, axis=0)


# In[31]:


sum(train['passenger_count']>6)


# In[32]:


plt.figure(figsize=(15,7))
sns.countplot("passenger_count", data=train)
#plot after removing outliers


# In[33]:


train.isnull().sum()


# In[34]:


sum(train['pickup_longitude']>180)
sum(train['pickup_longitude']<-180)


# In[35]:


sum(train['pickup_longitude']>90)


# In[36]:


sum(train['pickup_longitude']<-90)


# In[37]:


for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    print(i,'equal to 0={}'.format(sum(train[i]==0)))


# In[38]:


train=train.drop(train[train['pickup_latitude']==0].index,axis=0)


# In[39]:


train=train.drop(train[train['pickup_longitude']==0].index,axis=0)


# In[40]:


train=train.drop(train[train['dropoff_longitude']==0].index,axis=0)


# In[41]:


train=train.drop(train[train['dropoff_latitude']==0].index,axis=0)


# In[42]:


for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    print(i,'equal to 0={}'.format(sum(train[i]==0)))


# In[43]:


train.shape


# In[44]:


train.isnull().sum()


# In[45]:


#for category variables we impute with mode
train['passenger_count'] = train['passenger_count'].fillna(int(train['passenger_count'].mode()))


# In[46]:


train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)


# In[47]:


train.isnull().sum()


# In[48]:


train.shape


# In[49]:


train[train['fare_amount']<60]['fare_amount'].hist(bins=30, figsize=(14,3))
plt.xlabel('fare')
plt.title('Histogram')


# In[50]:


plt.figure(figsize=(14,5))
sns.countplot(x='passenger_count', data=train)


# In[51]:


train.describe()


# In[52]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# In[53]:


train.isnull().sum()


# In[54]:


test.isnull().sum()


# In[55]:


train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# In[56]:



train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[57]:


train.info()


# In[58]:


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],format= "%Y-%m-%d %H:%M:%S UTC")


# In[59]:


test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[60]:


test.info()


# In[61]:


test.head()


# In[62]:


train.head()


# In[63]:


plt.scatter(x=train['year'], y=train['fare_amount'], s=10)
plt.xlabel('Year')
plt.ylabel('Fare_amount')
plt.show()


# In[64]:


sns.countplot(train['year'])


# In[65]:


plt.scatter(x=train['Month'], y=train['fare_amount'], s=10)
plt.xlabel('Month')
plt.ylabel('Fare_amount')
plt.show()


# In[66]:


sns.countplot(train['Month'])


# In[67]:


plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare_amount')
plt.show()


# In[68]:


sns.countplot(train['Day'])


# In[69]:


plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare_amount')
plt.show()


# In[70]:


plt.figure(figsize=(30,10))

sns.countplot(train['Hour'])


# # Calculating Distance

# In[71]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min


# In[72]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[74]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[75]:


train.head()


# In[76]:


test.head()


# In[77]:


train['distance'].sort_values(ascending=False)


# In[78]:


sum(train['distance']>200)


# In[79]:


train=train.drop(train[train['distance']>200].index,axis=0)


# In[80]:



train = train.drop(train[train['distance'].isnull()].index, axis=0)


# In[81]:


train = train.drop(train[train['distance']== 0].index, axis=0)
train.shape


# In[82]:


print(train.isnull().sum())


# In[84]:


train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)
train = train.drop(train[train['year'].isnull()].index, axis=0)
train = train.drop(train[train['Month'].isnull()].index, axis=0)
train = train.drop(train[train['Date'].isnull()].index, axis=0)
train = train.drop(train[train['Day'].isnull()].index, axis=0)
train = train.drop(train[train['Hour'].isnull()].index, axis=0)
train = train.drop(train[train['Minute'].isnull()].index, axis=0)


# In[85]:


# realationship between fare and distance
plt.figure(figsize=(20,10))
plt.scatter(x="distance",y="fare_amount", data=train,color='blue')
plt.xlabel('distance')
plt.ylabel('fare')
plt.show()


# In[86]:


for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='grey')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[87]:



#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# In[88]:



for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[89]:


sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[90]:


test['distance'] = np.log1p(test['distance'])


# In[91]:


sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[92]:


numerical_val=['fare_amount','Date','distance','Hour','Day','passenger_count','year']


# In[93]:


#FEATURE SELECTION     #### FILTER METHOD ####    ## pearson correlation plot ##
train_corr=train.loc[:,numerical_val]
f, ax = plt.subplots(figsize=(7, 5))
correlation_matrix=train_corr.corr()
#correlation plot
sns.heatmap(correlation_matrix,mask=np.zeros_like(correlation_matrix,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax).get_figure().savefig('pythonheat_map.png')


# In[95]:


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(drop, axis = 1)


# In[96]:


train.head()


# In[97]:


drop_test = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
test = test.drop(drop_test, axis = 1)


# In[98]:


test.head()


# In[99]:


X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[100]:


print(X_train.shape)
print(X_test.shape)


# # LR MODEL

# In[128]:


LR = LinearRegression().fit(X_train , y_train)


# In[130]:


train_LR = LR.predict(X_train)


# In[132]:


test_LR = LR.predict(X_test)


# In[133]:



RMSE_train_LR= np.sqrt(mean_squared_error(y_train, train_LR))


# In[134]:


RMSE_test_LR = np.sqrt(mean_squared_error(y_test, test_LR))


# In[135]:


print("RMSE train = "+str(RMSE_train_LR))
print("RMSE test= "+str(RMSE_test_LR))


# In[136]:


from sklearn.metrics import r2_score
r2_score(y_train, train_LR)


# In[137]:


r2_score(y_test, test_LR)


# # DT Model

# In[138]:


DT_tree = DecisionTreeRegressor(max_depth = 3).fit(X_train,y_train)


# In[139]:


test_DT = DT_tree.predict(X_test)


# In[140]:


train_DT = DT_tree.predict(X_train)


# In[141]:


RMSE_train_DT = np.sqrt(mean_squared_error(y_train, train_DT))


# In[142]:


RMSE_test_DT = np.sqrt(mean_squared_error(y_test, test_DT))


# In[143]:


print("RMSE train= "+str(RMSE_train_DT))
print("RMSE test= "+str(RMSE_test_DT))


# In[144]:


r2_score(y_train, train_DT)


# In[145]:


r2_score(y_test, test_DT)


# # RF Model

# In[148]:


from sklearn.ensemble import RandomForestRegressor


# In[149]:


RF = RandomForestRegressor(n_estimators = 120).fit(X_train,y_train)


# In[151]:


train_RF = RF.predict(X_train)
test_RF = RF.predict(X_test)


# In[152]:


RMSE_train_RF = np.sqrt(mean_squared_error(y_train, train_RF))


# In[153]:


RMSE_test_RF = np.sqrt(mean_squared_error(y_test, test_RF))


# In[156]:


print("RMSE train ="+str(RMSE_train_RF))
print("RMSE test  = "+str(RMSE_test_RF))


# In[157]:


r2_score(y_train, train_RF)


# In[158]:


r2_score(y_test, test_RF)


# # GB Model

# In[159]:


from sklearn.ensemble import GradientBoostingRegressor


# In[161]:


GB = GradientBoostingRegressor().fit(X_train, y_train)


# In[166]:



train_GB = fit_GB.predict(X_train)
test_GB = fit_GB.predict(X_test)


# In[167]:


RMSE_train_GB = np.sqrt(mean_squared_error(y_train, train_GB))


# In[168]:


RMSE_test_GB = np.sqrt(mean_squared_error(y_test, test_GB))


# In[169]:


print("RMSE train = "+str(RMSE_train_GB))
print("RMSE test = "+str(RMSE_test_GB))


# In[170]:


r2_score(y_test, test_GB)


# In[171]:


r2_score(y_train, train_GB)


# # Random Search CV Hypertuning
# 

# In[173]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV


# # Random Search CV on Random Forest Model
# 

# In[174]:



RRF = RandomForestRegressor(random_state = 30)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_rf = RandomizedSearchCV(RRF, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=30)
randomcv_rf = randomcv_rf.fit(X_train,y_train)
predictions_RRF = randomcv_rf.predict(X_test)

view_best_params_RRF = randomcv_rf.best_params_

best_model = randomcv_rf.best_estimator_

predictions_RRF = best_model.predict(X_test)

#R^2
RRF_r2 = r2_score(y_test, predictions_RRF)
#RMSE
RRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_RRF))

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RRF)
print('R-squared = {:0.2}.'.format(RRF_r2))
print('RMSE = ',RRF_rmse)


# In[175]:


gb = GradientBoostingRegressor(random_state = 34)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())


# # Random Search CV on gradient boosting model
# 

# In[176]:



gb = GradientBoostingRegressor(random_state = 30)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_gb = RandomizedSearchCV(gb, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_gb = randomcv_gb.fit(X_train,y_train)
predictions_gb = randomcv_gb.predict(X_test)

view_best_params_gb = randomcv_gb.best_params_

best_model = randomcv_gb.best_estimator_

predictions_gb = best_model.predict(X_test)

#R^2
gb_r2 = r2_score(y_test, predictions_gb)
#Calculating RMSE
gb_rmse = np.sqrt(mean_squared_error(y_test,predictions_gb))

print('Random Search CV Gradient Boosting Model Performance:')
print('Best Parameters = ',view_best_params_gb)
print('R-squared = {:0.2}.'.format(gb_r2))
print('RMSE = ', gb_rmse)


# # Grid Search CV on RM model
# 

# In[182]:


from sklearn.model_selection import GridSearchCV    
## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 30)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF = gridcv_rf.predict(X_test)

#R^2
GRF_r2 = r2_score(y_test, predictions_GRF)
#Calculating RMSE
GRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_GRF))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_GRF)
print('R-squared = {:0.2}.'.format(GRF_r2))
print('RMSE = ',(GRF_rmse))


# # Grid Search CV on gradient boosting model
# 

# In[177]:


gb = GradientBoostingRegressor(random_state = 30)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

gridcv_gb = GridSearchCV(gb, param_grid = grid_search, cv = 5)
gridcv_gb = gridcv_gb.fit(X_train,y_train)
view_best_params_Ggb = gridcv_gb.best_params_

predictions_Ggb = gridcv_gb.predict(X_test)

#R^2
Ggb_r2 = r2_score(y_test, predictions_Ggb)
#Calculating RMSE
Ggb_rmse = np.sqrt(mean_squared_error(y_test,predictions_Ggb))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',view_best_params_Ggb)
print('R-squared = {:0.2}.'.format(Ggb_r2))
print('RMSE = ',(Ggb_rmse))


# # GB model

# In[179]:


regr = GradientBoostingRegressor(random_state = 30)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

predictions_GRF_test_Df = gridcv_rf.predict(test)


# # Prediction using GB model

# In[190]:


predictions_GRF_test_Df


# In[191]:


test.head()


# In[192]:


test.to_csv('test.csv')


# In[195]:


df=pd.read_csv(r"D:\Trim 5\ML\Supervised\test.csv")


# In[199]:


# realationship between fare and distance
plt.figure(figsize=(10,10))
plt.scatter(x="distance",y="Predicted_fare", data=df,color='blue')
plt.xlabel('distance')
plt.ylabel('fare')
plt.show()


# In[ ]:




