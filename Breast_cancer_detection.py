import numpy as np
import sklearn.datasets

breast_cancer = sklearn.datasets.load_breast_cancer()
#print(breast_cancer)

diabetes=sklearn.datasets.load_diabetes()
print(diabetes)

X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)
print(X.shape)
print(Y.shape)

import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.head()
data.describe()

print(data['class'].value_counts())

data.groupby('class').mean()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train , Y_test = train_test_split(X ,Y)

print(Y.shape , Y_train.shape , Y_test.shape)
X_train, X_test, Y_train , Y_test = train_test_split(X ,Y , test_size  = 0.1)
print(Y.shape , Y_train.shape , Y_test.shape)
print(Y.mean() , Y_train.mean() , Y_test.mean())
X_train, X_test, Y_train , Y_test = train_test_split(X ,Y , test_size  = 0.1, stratify = Y)
print(Y.mean() , Y_train.mean() , Y_test.mean())

print(X.mean(), X_train.mean(),X_test.mean())

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()

classifier.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score

prediction_on_training_data=classifier.predict(X_train)
accuracy_on_train_data = accuracy_score(Y_train , prediction_on_training_data)
print('accuracy on prediction data: ',accuracy_on_train_data)

prediction_on_test_data=classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test , prediction_on_test_data)
print(accuracy_on_test_data)

# predicting the model using KNN
#from sklearn.neighbors import KNeighborsClassifier
#nei = KNeighborsClassifier(n_neighbors=3)

#nei.fit(X_train,Y_train)
#from sklearn.metrics import accuracy_score

#prediction_on_training_data_using_Knn = nei.predict(X_train)
#accuracy_on_train_data_using_knn = accuracy_score(Y_train, prediction_on_training_data_using_Knn)
#print("accuraacy on traing data: ", accuracy_on_train_data_using_knn)

#prediction_on_test_data_using_knn = nei.predict(X_test)
#accuracy_on_test_data_using_knn = accuracy_score(Y_test , prediction_on_test_data_using_knn)

#print("accuracy on test data: ", accuracy_on_test_data_using_knn)

inptu_data = (13.71,20.83,90.2,577.9,0.1189,0.1645,0.09366,0.05985,0.2196,0.07451,0.5835,1.377,3.856,50.96,0.008805,0.03029,0.02488,0.01448,0.01486,0.005412,17.06,28.14,110.6,897,0.1654,0.3682,0.2678,0.1556,0.3196,0.1151)

input_data_as_numpy_array = np.asarray(inptu_data)
print(input_data_as_numpy_array)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)

prediction_actual = nei.predict(input_data_reshaped)
print(prediction_actual)
print('breast canser type based on prediction index: ', breast_cancer.target_names[0])
if prediction_actual[0]==0:
  print('breast canser type based on prediction index: ', breast_cancer.target_names[0])
else:
  print('breast canser type based on prediction index: ', breast_cancer.target_names[1])
