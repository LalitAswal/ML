import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r"C:\Users\aswal\Desktop\LDA\currency.csv")
#print(dataset)
df=pd.DataFrame(data=dataset)
x=df["YEAR"]
y=df["1 USD To INR"]
print(x,y)
#plt.scatter(x,y)
#
print(x)

#plt.show()
from scipy import stats
slope,intercept,r_value,p_value,std_err=stats.linregress(x,y)

# R value( accuracy)
print(r_value**2)


# function to create prediction/best fit line


def predict(y):
          return slope*y+intercept

x=np.array(x).reshape(-1,1)
fitLine=predict(x)
plt.scatter(x,y)
plt.plot(x,fitLine,c='r')
plt.show()


o=LinearRegression()
lm=o.fit(x,y)
print(lm)
years=[]

for x in range(3):
          a=float(input("enter the years :"))
          years.append(a)
years=np.array(years).reshape(-1,1)

print(lm.predict(years))

