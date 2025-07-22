

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target


model = RandomForestClassifier()
model.fit(X, y)


joblib.dump(model, 'model.pkl')
print("✅ Model saved successfully as 'model.pkl'")
