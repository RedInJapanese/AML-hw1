import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train  = pd.read_csv("train.csv")


#cont feature
cf = "GrLivArea"


plt.figure(figsize=(8, 5))
sns.histplot(train[cf], bins=40, kde=True)
plt.title(f"GRLivArea")
plt.xlabel(cf)
plt.ylabel("Count")
plt.show()

#categorical feature
cf2 = "Neighborhood"  

plt.figure(figsize=(12, 6))
sns.countplot(y=cf2, data=train, order=train[cf2].value_counts().index)
plt.title(f"Distribution of {cf2}")
plt.xlabel("Count")
plt.ylabel(cf2)
plt.show()




#part 1 question 4

ohe_feature = "CentralAir"

plt.figure(figsize=(5,4))
sns.countplot(x=ohe_feature, data=train)
plt.title(f"Original Distribution")
plt.xlabel(ohe_feature)
plt.ylabel("Count")
plt.show()


#OHE
train_oh = pd.get_dummies(train[ohe_feature], prefix=ohe_feature)
train_encoded = pd.concat([train[[ohe_feature]], train_oh], axis=1)

train_oh.sum().plot(kind="bar", figsize=(6,4))
plt.title(f"One-Hot Encoding")
plt.ylabel("Count")
plt.show()

