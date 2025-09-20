import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#training and test sets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_id = test["PassengerId"]

features = ['P class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



