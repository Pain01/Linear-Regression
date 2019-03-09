import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
print(dataset.head())
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]


from sklearn.cross_validation import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

r = LinearRegression()
r.fit(X_train,y_train)


y_pred = r.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
print(X)
X_op = X[:, [0,1]]

r_OLS = sm.OLS(endog = y, exog = X_op).fit()
print(r_OLS.summary())





