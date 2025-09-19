import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train  = pd.read_csv("train.csv")


# #cont feature
# cf = "GrLivArea"


# plt.figure(figsize=(8, 5))
# sns.histplot(train[cf], bins=40, kde=True)
# plt.title(f"GRLivArea")
# plt.xlabel(cf)
# plt.ylabel("Count")
# plt.show()

# #categorical feature
# cf2 = "Neighborhood"  

# plt.figure(figsize=(12, 6))
# sns.countplot(y=cf2, data=train, order=train[cf2].value_counts().index)
# plt.title(f"Distribution of {cf2}")
# plt.xlabel("Count")
# plt.ylabel(cf2)
# plt.show()




# #part 1 question 4

# ohe_feature = "Neighborhood"

# plt.figure(figsize=(12,6))
# sns.countplot(y=ohe_feature, data=train, order=train[ohe_feature].value_counts().index)
# plt.title(f"Original ")
# plt.xlabel("Count")
# plt.ylabel(ohe_feature)
# plt.show()

# # ohe
# train_onehot = pd.get_dummies(train[ohe_feature], prefix=ohe_feature)

# train_encoded = pd.concat([train[[ohe_feature]], train_onehot], axis=1)


# train_onehot.sum().sort_values(ascending=False).plot(kind="bar", figsize=(14,6))
# plt.title(f"One-Hot Encode")
# plt.ylabel("Count")
# plt.show()


# # print(train_encoded.head())





# Question 5

features = ["GrLivArea", "OverallQual", "CentralAir"]
target = "SalePrice"


#OHE
train = pd.get_dummies(train, columns=["CentralAir"], drop_first=True)

# Build X (features) and y (target)
x = train[["GrLivArea", "OverallQual", "CentralAir_Y"]].astype(float).values
y = train[target].values.reshape(-1, 1)

x = np.hstack([np.ones((x.shape[0], 1)), x])  # samples, featuers + 1

# OLS = (X^T X)^(-1) X^T y
xtx = x.T @ x
xtx_inverse = np.linalg.inv(xtx)
xty = x.T @ y
beta = xtx_inverse @ xty  

# regression coefficient ^^^^ 
