# -*- coding: utf-8 -*-
"""home-loan-approval-prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HzWI1GiXTxN4DsIbYdbOzmCz1fHfdEoU

#HOME LOAN APPROVAL (PREDICTIVE ANALYTICS)
*   Nama: Iva Raudyatuzzahra
*   Dataset : https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval

#Data Collection

Import library & packages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Load dataset"""

data = pd.read_csv('/content/loan_sanction_train.csv')
data

"""#Exploratory Data Analysis

##Variable Description
"""

data.info()

"""##Drop Loan_ID"""

data.drop(['Loan_ID'], axis=1, inplace = True)
data

"""##Statistic Description"""

data.describe()

"""##Checking missing values"""

data.isnull().sum()

"""##Handling Missing Values
Karena jumlah data yang terbatas, data yg mengandung missing value tidak akan di-drop melainkan akan disubtitusi dengan nilai yg sering muncul (modus) kecuali untuk kolom Loan_Amount akan disubtitusi dengan nilai rata-ratanya

###Mencari common answer atau modus data
"""

import statistics
print('the common answer for Gender column:', statistics.mode(data['Gender']))
print('the common answer for Married column:', statistics.mode(data['Married']))
print('the common answer for Dependents column:', statistics.mode(data['Dependents']))
print('the common answer for Self_Employed column:', statistics.mode(data['Self_Employed']))
print('the common answer for Loan_Amount_Term column:', statistics.mode(data['Loan_Amount_Term']))
print('the common answer for Credit_History column:', statistics.mode(data['Credit_History']))

"""###Impute missing values"""

new_data = data.copy()

new_data['Gender'].fillna(new_data['Gender'].value_counts().idxmax(), inplace=True)
new_data['Married'].fillna(new_data['Married'].value_counts().idxmax(), inplace=True)
new_data['Dependents'].fillna(new_data['Dependents'].value_counts().idxmax(), inplace=True)
new_data['Self_Employed'].fillna(new_data['Self_Employed'].value_counts().idxmax(), inplace=True)
new_data["LoanAmount"].fillna(new_data["LoanAmount"].mean(skipna=True), inplace=True)
new_data['Loan_Amount_Term'].fillna(new_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
new_data['Credit_History'].fillna(new_data['Credit_History'].value_counts().idxmax(), inplace=True)

"""
###Recheck missing values"""

new_data.isnull().sum()

print("Jumlah Pinjaman yang disetujui dan tidak :")
print(data['Loan_Status'].value_counts())
sns.countplot(x='Loan_Status', data=data, palette = 'Set2')

"""#Data Preparation

##Encoding

Mengubah kolom kategorikal (Gender, Married, Dependents, Education, Self_Employed, Property_Area) menjadi numerik
"""

new_data1 = new_data.copy()

gender_dummies = pd.get_dummies(new_data1['Gender'],prefix="Gender",drop_first=True)
new_data1 = new_data1.drop('Gender',axis = 1)
new_data1 = new_data1.join(gender_dummies)
new_data1['Married'] = new_data1['Married'].map({'Yes':1, 'No':2})
new_data1['Self_Employed'] = new_data1['Self_Employed'].map({'Yes':1, 'No':2})
new_data1['Education'] = new_data1['Education'].map({'Graduate':1, 'Not Graduate':2})
new_data1['Loan_Status'] = new_data1['Loan_Status'].map({'Y':1, 'N':2})
new_data1['Dependents'] = new_data1['Dependents'].map({'0': 0, '1':1, '2':2, '3+':3})
PA_dummies = pd.get_dummies(new_data1['Property_Area'],prefix="PA",drop_first=True)
new_data1 = new_data1.drop('Property_Area',axis = 1)
new_data1 = new_data1.join(PA_dummies)

new_data1.head()

"""##Split Data
Membagi dataset menjadi train dan test data
"""

X = new_data1.drop('Loan_Status',axis = 1)
y = new_data1['Loan_Status']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

"""#Model Development

Import library
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

"""membuat variabel"""

classifier = ('K-Nearest Neighbor', 'Random Forest', 'Gradient Boosting')
y_pos = np.arange(len(classifier))
score = []

"""##KNN"""

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
scores = cross_val_score(knn, X, y,cv=5)
score.append(scores.mean())
print('Akurasi model KNN adalah %.2f%%' %(scores.mean()*100))

"""##Random Forest"""

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X, y,cv=5)
score.append(scores.mean())
print('Akurasi model Random Forest adalah %.2f%%' %(scores.mean()*100))

"""##Gradient Boosting"""

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
scores = cross_val_score(gb, X, y,cv=5)
score.append(scores.mean())
print('Akurasi model Gradient Boosting adalah %.2f%%' %(scores.mean()*100))

"""#Evaluation"""

plt.barh(y_pos, score, align='center', alpha=0.5)
plt.yticks(y_pos, classifier)
plt.xlabel('Score')
plt.title('Classification Performance')
plt.show()

"""##MSE"""

from sklearn.metrics import mean_squared_error
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': rf, 'Boosting': gb}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

mse

"""##Prediction"""

prediksi = X_test.iloc[25:29].copy()
pred_dict = {'y_true':y_test[25:29]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)