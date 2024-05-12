# -*- coding: utf-8 -*-


                           #Electric Price Prediction

import pandas as pd
import numpy as np

#Download Data
data =pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv")
data.head()


#DateTime: Date and time of the record
#Holiday: contains the name of the holiday if the day is a national holiday
#HolidayFlag: contains 1 if it’s a bank holiday otherwise 0
#DayOfWeek: contains values between 0-6 where 0 is Monday
#WeekOfYear: week of the year
#Day: Day of the date
#Month: Month of the date
#Year: Year of the date
#PeriodOfDay: half-hour period of the day
#ForcastWindProduction: forecasted wind production
#SystemLoadEA forecasted national load
#SMPEA: forecasted price
#ORKTemperature: actual temperature measured
#ORKWindspeed: actual windspeed measured
#CO2Intensity: actual C02 intensity for the electricity produced
#ActualWindProduction: actual wind energy production
#SystemLoadEP2: actual national system load
#SMPEP2: the actual price of the electricity consumed (labels or values to be predicted)

#Let’s have a look at all the columns of this dataset:
data.info()    


#I can see that so many features with numerical values are string values in the dataset 
#and not integers or float values. So before moving further, 
#we have to convert these string values to float values:Numeric

data["ForecastWindProduction"]= pd.to_numeric(data["ForecastWindProduction"],errors= "coerce")
data["SystemLoadEA"]= pd.to_numeric(data["SystemLoadEA"],errors= "coerce")
data["SMPEA"]= pd.to_numeric(data["SMPEA"],errors= "coerce")
data["ORKTemperature"]= pd.to_numeric(data["ORKTemperature"],errors= "coerce")
data["ORKWindspeed"]= pd.to_numeric(data["ORKWindspeed"],errors= "coerce")
data["CO2Intensity"]= pd.to_numeric(data["CO2Intensity"],errors= "coerce")
data["ActualWindProduction"]= pd.to_numeric(data["ActualWindProduction"],errors= "coerce")
data["SystemLoadEP2"]= pd.to_numeric(data["SystemLoadEP2"],errors= "coerce")
data["SMPEP2"]= pd.to_numeric(data["SMPEP2"],errors= "coerce")

data.info()


#Now let’s have a look at whether this dataset contains any null values or not:
data.isnull().sum()

#So there are some columns with null values, I will drop all these rows containing
#null values from the dataset:
data = data.dropna()
data


#Now let’s have a look at the correlation between all the columns in the dataset:
import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include='number')
numeric_data.drop("HolidayFlag", axis=1, inplace=True)
correlations = numeric_data.corr()

plt.figure(figsize=(16,12))
plt.title('Correlation Matrix')
sns.heatmap(correlations,cmap="coolwarm",annot=True)
plt.show()



#Grafiksel Analiz - hedef değişken üzerinde etkisi
import matplotlib.pyplot as plt

# Grafik boyutunu ayarla
plt.figure(figsize=(15, 8))

# Scatter plot grafiğini çiz
plt.scatter(data['ForecastWindProduction'], data['SMPEP2'], label='Forecast Wind Production', alpha=0.5)
plt.scatter(data['SystemLoadEA'], data['SMPEP2'], label='System Load EA', alpha=0.5)
plt.scatter(data['ORKTemperature'], data['SMPEP2'], label='ORK Temperature', alpha=0.5)
plt.scatter(data['ORKWindspeed'], data['SMPEP2'], label='ORK Windspeed', alpha=0.5)
plt.scatter(data['SMPEA'], data['SMPEP2'], label='SMPEA', alpha=0.5)
plt.scatter(data['CO2Intensity'], data['SMPEP2'], label='CO2 Intensity', alpha=0.5)
plt.scatter(data['ActualWindProduction'], data['SMPEP2'], label='Actual Wind Production', alpha=0.5)
plt.scatter(data['SystemLoadEP2'], data['SMPEP2'], label='System Load EP2', alpha=0.5)
plt.xlabel('Feature Values')
plt.ylabel('SMPEP2')
plt.title('Relationship Between Features and SMPEP2')
plt.legend()
plt.show()



#Analysis of Variable SMPEP2
plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.title('SMPEP2 Distribuition')
sns.distplot(data['SMPEP2'])

plt.subplot(122)
g1 = plt.scatter(range(data.shape[0]), np.sort(data.SMPEP2.values))
g1= plt.title("SMPEP2 Curve Distribuition", fontsize=15)
g1 = plt.xlabel("")
g1 = plt.ylabel("SMPEP2", fontsize=12)
plt.subplots_adjust(wspace = 0.3, hspace = 0.5,
                    top = 0.9)
plt.show()



sns.histplot(data['ForecastWindProduction'], bins = 10, kde = True)
sns.histplot(data['SystemLoadEA'], bins = 10, kde = True)
sns.histplot(data['SMPEA'], bins = 10, kde = True)
sns.histplot(data['ORKWindspeed'], bins = 10, kde = True)
sns.histplot(data['CO2Intensity'], bins = 10, kde = True)
sns.histplot(data['ActualWindProduction'], bins = 10, kde = True)
sns.histplot(data['SystemLoadEA'], bins = 10, kde = True)



#Analysis of SystemLoadEA and SMPEP2
plt.figure(figsize=(12,6))
sns.boxplot( x=data['SystemLoadEA'], y=data['SMPEP2'] )

plt.title('Statistical Distribution of SystemLoadEA versus SMPEP2')
plt.show()



            # checking the count of unique specialization present in dataframe
            
data.Holiday.value_counts()

# count plot of unique categories in holiday

# create the copy of dataframe
datac = data.copy()
# count of unique categories in holiday
value_count = datac['Holiday'].value_counts()

def map_to_other_holiday(var):
    ''' if count of unique category is less than 10, replace the category as other '''
    if var in value_count[value_count<=40]:
        return 'other'
    else:
        return var
    
# apply the function to holiday to get the results    
data['Holiday'] = data.Holiday.apply(map_to_other_holiday)

# count plot of unique categories in holiday
plt.figure(figsize = (16, 8))
total = float(len(data))
ax = sns.countplot(x='Holiday',data=data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 1,
            '{:1.2f}%'.format((height/total) * 100),
            ha="center",fontsize=10) 
plt.xticks(rotation = 90)
plt.show()

            #average smpep2 by holiday and sort them in decreasing order
avg_sal_per_holiday = data.groupby('Holiday').agg(mean_Smpep2=("SMPEP2", 'mean')).sort_values(by='mean_Smpep2', ascending=False)

# barplot of mean salary and specialization
plt.figure(figsize = (12, 6))
sns.barplot(x = avg_sal_per_holiday.index,y = 'mean_Smpep2',data = avg_sal_per_holiday,palette='rocket')
plt.xticks(rotation = 90)
plt.show()





##############################################################################################

                        # Electricity Price Prediction Model
     
#Now let’s move to the task of training an electricity price prediction model. 
#Here I will first add all the important features to x and the target column to y, 
#and then I will split the data into training and test sets:                 
     
x = data[["Day","Month","ForecastWindProduction","SystemLoadEA","SMPEA",
        "ORKTemperature","ORKWindspeed","CO2Intensity","ActualWindProduction",
        "SystemLoadEP2"]]
y = data["SMPEP2"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#I will choose the Random Forest regression algorithm to train the electricity price prediction model:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)

#RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                      max_depth=None, max_features='auto', max_leaf_nodes=None,
#                     max_samples=None, min_impurity_decrease=0.0,
#                      min_impurity_split=None, min_samples_leaf=1,
#                      min_samples_split=2, min_weight_fraction_leaf=0.0,
#                     n_estimators=100, n_jobs=None, oob_score=False,
#                     random_state=None, verbose=0, warm_start=False)


#r2 score
from sklearn.metrics import r2_score
print("Random Forest R2 değeri:")
print(r2_score(y,model.predict(x)))



#accurcay score
accuracy_score=model.score(x_test,y_test)*100
accuracy_score


# Obtain the model's predictions
y_pred = model.predict(x_test)


# Calculate the mean square error between predictions and actual values
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error:", mse)
#403.95 bir hata oranı, modelin tahminlerinin ortalama olarak gerçek değerlerden
#ortalama olarak yaklaşık 20 birim uzakta olduğunu gösterir.


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)



#Modelimizde değişkenlerin önemi - Feature Importance Analizi
importance = model.feature_importances_
feature_importance_data = {
    "Feature": x.columns,
    "Importance": importance
}
feature_importance = pd.DataFrame(feature_importance_data)
print(feature_importance)



#Now let’s input all the values of the necessary features that we used to train the 
#model and have a look at the price of the electricity predicted by the model:

#Features = [["Day","Month","ForecastWindProduction","SystemLoadEA","SMPEA",
#        "ORKTemperature","ORKWindspeed","CO2Intensity","ActualWindProduction",
#        "SystemLoadEP2"]]

features = np.array([[15,6,61.2,3567.2,68.21,14,35.1,700,54.0,3000]])
model.predict(features)

#So this is how you can train a machine learning model to predict the prices of electricity.


##############################################################################################

                      #Grid Search ile Random Forest Modeli:
                          
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# RandomForestRegressor modelini oluştur
modelg = RandomForestRegressor()

# Parametre gridini belirt
param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [1, 2, 3, 4, 5, 6, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV'yi oluştur
grid_search = GridSearchCV(estimator=modelg, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=10, n_jobs=-1)

# RandomForestRegressor modelini eğit
modelg.fit(x_train, y_train)

# Grid search uygula
grid_search.fit(x_train, y_train)

# En iyi parametreleri al
best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=30, min_samples_split=2, n_estimators=100)
model.fit(x_train,y_train)




from sklearn.metrics import r2_score
print("Random Forest R2 değeri:")
print(r2_score(y,model.predict(x)))




accuracy_score=model.score(x_test,y_test)*100
accuracy_score


features = np.array([[15,6,61.2,3567.2,68.21,14,35.1,700,54.0,3000]])
model.predict(features)

##############################################################################################


                     #Support Vector Regression Tahmin Yöntemi
 
#Now let’s move to the task of training an electricity price prediction model. 
#Here I will first add all the important features to x and the target column to y, 
#and then I will split the data into training and test sets:                 
     
x = data[["Day","Month","ForecastWindProduction","SystemLoadEA","SMPEA",
        "ORKTemperature","ORKWindspeed","CO2Intensity","ActualWindProduction",
        "SystemLoadEP2"]]
y = data["SMPEP2"]    
 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#Dataframe slicelearını Numpy Array dönüşümü
X = x.values
Y = y.values
    
#Verileri Ölçekleme:(Standartlaştırma) Sayıların dönüşümünü istiyoruz.

from sklearn.preprocessing import StandardScaler    

sc1= StandardScaler()
x_ölçekli = sc1.fit_transform(X)

sc2= StandardScaler()
y_ölçekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))    
 
#Regresyon algoritmasını Modelini oluşturma:
from sklearn.svm import SVR
svr_reg =SVR(kernel='rbf')
svr_reg.fit(x_ölçekli,y_ölçekli)    
 
plt.scatter(x_ölçekli,y_ölçekli,color='red')
plt.plot(x_ölçekli,svr_reg.predict(x_ölçekli)) 
 
from sklearn.metrics import r2_score
print("Support Vector Regression R2 değeri:")
print(r2_score(y_ölçekli,svr_reg.predict(x_ölçekli))) 




from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred = svr_reg.predict(x_ölçekli)

mae = mean_absolute_error(y_ölçekli, y_pred)
mse = mean_squared_error(y_ölçekli, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)



features = np.array([[15,6,61.2,3567.2,68.21,14,35.1,700,54.0,3000]])
svr_reg.predict(features)    
 
##############################################################################################   

##############################################################################################3
       
                                  #Görselleştirmeler

#Scatter - DateTime&Price

import plotly.graph_objs as go
import plotly.offline as pyo
import pandas as pd

trace =go.Scatter(x =data["DateTime"],y=data["SMPEP2"],mode="lines",name="Electricity Price")
layout = go.Layout(title="Electricity Price Over Time",xaxis=dict(title="DateTime"),
                   yaxis=dict(title='Price'))

fig = go.Figure(data=[trace],layout=layout)
pyo.plot(fig,filename="electricty_price.html")


##########################################

#Bar - Holiday&Date

import plotly.graph_objs as go
import plotly.io as pyo
import plotly.offline as pyo

trace = go.Bar(x=data["Holiday"], y=data["SMPEP2"], name="Electricity Price")
layout = go.Layout(title="Electricity Price by Holiday", xaxis=dict(title="Holiday"), yaxis=dict(title='Price'))
fig = go.Figure(data=[trace], layout=layout)

# HTML dosyasına kaydetme
pyo.plot(fig, filename="electricity_price_holiday.html")














