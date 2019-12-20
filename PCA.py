import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



a=pd.read_csv(r'C:\Users\aswal\Desktop\housing_train.csv')
df=pd.DataFrame(data=a)
X=df[['OverallQual','OverallCond','GrLivArea','FullBath','HalfBath','BedroomAbvGr','GarageCars','GarageArea']].values
print(X)
y=df[['SalePrice']].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=70)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


sc = StandardScaler() 
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA 
  
pca = PCA(n_components = 2) 
  
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
  
classifier = RandomForestClassifier(max_depth=2, random_state=70)

classifier.fit(X_train, y_train.ravel())
y_pred = classifier.predict(X_test)
print(y_pred) 

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))



