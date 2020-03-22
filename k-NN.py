# Polynomial Regression

# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('train.csv')

y1 = dataset.iloc[:,-2].values
y2 = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

dataset=dataset.apply(LabelEncoder().fit_transform)
X = dataset.iloc[:,1:5].values
# Feature Scaling
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
from sklearn.metrics import accuracy_score 
print( 'Accuracy Score confirmed cases :',accuracy_score(y_test,y_pred1)*100) 




from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
classifier.fit(X_train2, y_train2)


# Predicting the Test set results
y_pred2 = classifier.predict(X_test2)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred2)
from sklearn.metrics import accuracy_score 
print( 'Accuracy Score fatality:',accuracy_score(y_test2,y_pred2)*100) 