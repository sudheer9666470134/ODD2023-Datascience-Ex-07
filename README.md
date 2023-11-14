# Ex-07-Feature-Selection
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
- STEP 1
Read the given Data
- STEP 2
Clean the Data Set using Data Cleaning Process
- STEP 3
Apply Feature selection techniques to all the features of the data set
- STEP 4
Save the data to the file

# PROGRAM

- <B>DATA PREPROCESSING BEFORE FEATURE SELECTION:</B>
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/ec7095e1-2eb1-4b6e-8f1e-020490c45f18">

- <B>CHECKING NULL VALUES:</B>
```python
df.isnull().sum()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/9a9c0052-7799-4c3f-b776-3a1c3a74a9e4">

- <B>DROPPING UNWANTED DATAS:</B>
```python
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/bf454256-8858-4cc5-b117-3cbe8397f0c4">

- <B>DATA CLEANING:</B>
```python
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
<img src="(https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/84a351dc-54ce-4459-98af-39b8ba5e28ce)">

- <B>REMOVING OUTLIERS:</B>
  - Before
```python
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/4403a339-4e7f-446d-a4e2-8c842945f3c2">
 
- 
  - After
```python
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/3ca363fe-1a63-451b-8d4c-df0829b36154">


- <B>_FEATURE SELECTION:_</B>
```python
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/dc5f747d-28e0-4d00-9e11-4dbaefbebb82">

```python
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/0cef6fb3-a7a4-43e0-9da1-1248257de5a8">

```python
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/5d2c7bfa-9ba3-4506-b1b2-c6455b17687f">


```python
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/d64bf214-8694-4e1b-93a4-7804081638ab">


```python
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 
```

- <B>_FILTER METHOD:_</B>
```python
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/5a3f1594-b1bf-4b50-9644-d7373e2af1fc">

- <B>_HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:_</B>
```python
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/4a5b1431-bac4-4b25-81f5-272a896d6c88">

- <B>_BACKWARD ELIMINATION:_</B>
```python
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/2e3cf737-9f3f-481b-94ca-bd8898439111">


- <B>_OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:_</B>
```python
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/2080ea66-3b61-4afa-a88e-63729ed5adc6">

- <B>_FINAL SET OF FEATURE:_</B>
```python
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/2eb3ff61-48d9-4605-89c7-b83ea8a608e4">

- <B>_EMBEDDED METHOD:_</B>
```python
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
<img src="https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex-07/assets/119393540/aeac1e14-75a9-4d1e-b2f9-7079ac3382cf)">

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
