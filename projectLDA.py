#Multiclass classification with linear Discriminant Analysis
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

lda = LDA()
X_train = lda.fit_transform(X_train, y_train.ravel())
X_test = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=2, random_state=70)

classifier.fit(X_train, y_train.ravel())
y_pred = classifier.predict(X_test)
print(y_pred) 

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))

models=[]
models.append(('LDA', LDA()))
models.append(('KNN', KNeighborsClassifier()))
results =[]
names = []
              
for name,model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        cv_results = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
X=X.flatten()        
X.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
