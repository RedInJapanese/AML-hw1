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