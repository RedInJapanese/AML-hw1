import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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
x["Embarked"] = x["Embarked"].fillna(x["Embarked"].median())
x["Fare"] = x["Fare"].fillna(x["Fare"].median())


#encode categorical features
x["Sex"] = X["Sex"].map({"male": 0, "female": 1})
# OHEing embarked
x = pd.get_dummies(x, columns=["Embarked"], drop_first=True)


