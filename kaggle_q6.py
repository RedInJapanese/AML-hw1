import pandas as pd
import numpy as np

 #load the datasest
 #add test id to the csv 
 #isolate sales price and id 
 #
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["Id"]

#isolate sales price and ID
y = train["SalePrice"].astype(float).values.reshape(-1, 1) #takes sales price and converts it to float and column vector
train_features = train.drop(columns=["SalePrice", "Id"]) #remove sales price and the id from the training data
test_features = test.drop(columns=["Id"]) #remove id(kaggle doesnt show sales price)


#concatenate the training and test features
submission = pd.concat([train_features, test_features], axis=0)

#fill missing values
for col in submission.columns:
    if submission[col].dtype == "object":
        print(submission[col])
        submission[col] = submission[col].fillna(submission[col].mode()[0])
    else:
        submission[col] = submission[col].fillna(submission[col].median())

submission = pd.get_dummies(submission, drop_first=True)

x_train = submission.iloc[:len(train), :].astype(float).values
x_test = submission.iloc[len(train):, :].astype(float).values


# OLS
xtx = x_train.T @ x_train
xtx_inv = np.linalg.pinv(xtx)  # safer than inv()
xty = x_train.T @ y
beta = xtx_inv @ xty
print(beta)
y_pred = x_test @ beta
print(y_pred)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_pred.flatten()
})

submission.to_csv("submission.csv", index=False)