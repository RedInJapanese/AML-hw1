import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#training and test sets
train = pd.read_csv("train2.csv")
test = pd.read_csv("test2.csv")

test_id = test["PassengerId"]

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


#features in x axis, survival in the y axis
x = train[features].copy()
y = train["Survived"]

#missing ages need to be filled with mediasn, same with embarked and fare
x["Age"] = x["Age"].fillna(x["Age"].median())
x["Embarked"] = x["Embarked"].fillna(x["Embarked"].mode()[0])
x["Fare"] = x["Fare"].fillna(x["Fare"].median())


#encode categorical features
x["Sex"] = x["Sex"].map({"male": 0, "female": 1})
# OHEing embarked
x = pd.get_dummies(x, columns=["Embarked"], drop_first=True)


x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=200)    
model.fit(x_train, y_train)


y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
conf_mat = confusion_matrix(y_val, y_pred)


print("Validation Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)


x_test = test[features].copy()
x_test["Age"] = x_test["Age"].fillna(x_test["Age"].median())
x_test["Embarked"] = x_test["Embarked"].fillna(x_test["Embarked"].mode()[0])
x_test["Fare"] = x_test["Fare"].fillna(x_test["Fare"].median())
x_test["Sex"] = x_test["Sex"].map({"male": 0, "female": 1})
x_test = pd.get_dummies(x_test, columns=["Embarked"], drop_first=True)

# Ensure test set has same columns as training set
missing_cols = set(x_train.columns) - set(x_test.columns)
for col in missing_cols:
    x_test[col] = 0
x_test = x_test[x_train.columns]  # reorder columns

test_pred = model.predict(x_test)
submission = pd.DataFrame({"PassengerId": test_id, "Survived": test_pred})
submission.to_csv("t_submission.csv", index=False)
print("Saved t_submission.csv")


