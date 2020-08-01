import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# set matplotlib theme
sns.set()

# load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
df.to_csv("df.csv")

# plotting
sns.countplot(x="Sex", data=df, hue="Survived")
# sns.countplot(x="Survived", hue="AgeCat", data=df)
# sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=df)
# sns.catplot("Survived", "Fare", kind="bar", data=df)
# sns.distplot(df["Fare"])
# sns.countplot(x="Fare", hue="Survived", data=df)
plt.show()
