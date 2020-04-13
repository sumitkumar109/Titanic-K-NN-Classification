# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating Dataset
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:,[2,4]]
y = dataset.iloc[:,1]


# Taking care of Missing Data
x.fillna(x.median(), inplace=True)

# Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0,1])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

# Fitting data to K-NN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x,y)


#*************************************** TEST DATA *********************************************************


# Importing Test Dataset
test_dataset = pd.read_csv('test.csv')
x_test = test_dataset.iloc[:,[1,3]]

# Taking care of Missing Data
x_test.fillna(x_test.median(), inplace=True)

# Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0,1])], remainder='passthrough')
x_test = np.array(ct.fit_transform(x_test), dtype=np.float)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_test = sc_x.fit_transform(x_test)

# Prediction
y_pred = classifier.predict(x_test)

# Confusion Matrix
y_test = pd.read_csv('gender_submission-cpy.csv')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)








