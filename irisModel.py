import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
from pydantic import BaseModel


class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisMachineLearning:

    def __init__(self):
        self.iris_df = pd.read_csv('iris.csv')
        self.rfc_fname = 'iris_rfc.pkl'
        try:
            self.model_rfc = joblib.load(self.rfc_fname)
        except Exception as _:
            self.model_rfc = self.rfc_train()
            joblib.dump(self.model_rfc, self.rfc_fname)
        return

    def rfc_train(self):
        X = self.iris_df.drop('species', axis=1)
        y = self.iris_df['species']
        rfc = RandomForestClassifier()
        model = rfc.fit(X, y)
        return model

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        x_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model_rfc.predict(x_new)
        print(f'RFC prediction={prediction}')
        probability = self.model_rfc.predict_proba(x_new).max()
        print(f'RFC prob={probability}')
        return prediction[0], probability
