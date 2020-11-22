import numpy as np
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("glass.csv")
X = df.iloc[:,:-1]
y = df.iloc[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

cls = RandomForestClassifier()
cls.fit(X, y)

prediction = cls.predict(X)
prediction_proba = cls.predict_proba(X)

print("Accuracy is : ", cls.score(X_test, y_test)*100, '%')

#save the model to the disk
filename = 'finalized_model.sav'
joblib.dump(cls, filename)