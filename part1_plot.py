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