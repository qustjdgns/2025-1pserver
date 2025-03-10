#2025.3.10
#프로젝트2 붓꽃분류기 만들기



import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_df = pd.read_csv('iris.csv')

# training
print(iris_df)

X = iris_df.drop('species', axis=1)
y = iris_df['species']
print(X)
print(y)
rfc = RandomForestClassifier()
model_rfc = rfc.fit(X, y)

kn = KNeighborsClassifier()
model_kn = kn.fit(X, y)

#모델 저장 및 활용
joblib.dump(model_rfc, 'model_rfc.pkl')
joblib.dump(model_kn, 'model_kn.pkl')

model_rfc = joblib.load('model_rfc.pkl')
model_kn = joblib.load('model_kn.pkl')

# 예측
X_new = np.array([[1, 4.2, 1.4, 7]])
#X_new = np.array([[3, 3, 3, 3]])
#KN prob=[[0.  0.8 0.2]]
#X_new = np.array([[1, 4.2, 1.4, 7]])
#KN prob=[[0.2 0.6 0.2]]
#X_new = np.array([[5.0, 3.4, 1.4, 0.2]])
# KN prob=[[1. 0. 0.]]
prediction = model_rfc.predict(X_new)
print(f'RFC prediction={prediction}')
probability = model_rfc.predict_proba(X_new)
print(f'RFC prob={probability}')

prediction = model_kn.predict(X_new)
print(f'KN prediction={prediction}')
probability = model_kn.predict_proba(X_new)
print(f'KN prob={probability}')