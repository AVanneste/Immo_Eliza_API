import imp
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle

df= pd.read_csv('coef.csv')

#for region based prediction
# df = df.loc[df['Region']=='Brussels'] #regions are; Brussels, Flanders, Wallonia 

#Creating new DFs interms of 'type of propery'
df_house = df.loc[(df['type of property']=='HOUSE')]
df_apartment = df.loc[(df['type of property']=='APARTMENT')]
df_others = df.loc[(df['type of property']=='OTHERS')]

#Seperation of numeric values and categorical values
numeric_data_df = df_house.select_dtypes(include=[np.number])
categorical_data_df = df_house.select_dtypes(exclude=[np.number])

#Setting the value for X and y
X= numeric_data_df.drop(['Price', 'post code', 'id', 'Unnamed: 0'], axis=1)
y = numeric_data_df[['Price']]
X.info()
y.info()

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)

#Fitting the Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()  
mlr.fit(X_train, y_train)
print(y_train)

#Intercept and Coefficients
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(X, mlr.coef_))

#Prediction of test set
y_pred_mlr= mlr.predict(X_test)
y_pred_mlr = y_pred_mlr.round(0)
#Predicted values of price
print(f"Prediction for train set: {y_pred_mlr}")

#Actual value and the predicted value of price
mlr_diff = pd.DataFrame()
mlr_diff['Actual value'] = y_test
mlr_diff['Predicted value'] = y_pred_mlr
mlr_diff.head()

#Checking all the predicted values are positive
for a in y_pred_mlr:
    if a<0:
        print('negative price??!!') 
    else:
        continue

#Calculating the metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(X,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(mlr, open(filename, 'wb'))

