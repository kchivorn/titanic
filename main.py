import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
print(df.head())
print(df.info())

df = df.drop(['Ticket','Cabin'], axis=1)
df = df.dropna()
print(df.head())
print(df.info())

fig = plt.figure(figsize=(18,6), dpi=120)
alpha=alpha_scatterplot =0.2
alpha_bar_chart = 0.55

ax1 = plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax1.set_xlim(-1,2)
plt.title("Distribution of Survival, (1 = Survived)")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)

plt.ylabel("Age")
plt.grid(b=True, which='major', axis='y')
plt.title("Survival by Age, (1 = Survived)")

ax3 = plt.subplot2grid((2,3), (0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=alpha_scatterplot)
ax3.set_ylim(-1, len(df.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0),colspan=2)
df.Age[df.Pclass == 1].plot(kind='kde')
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')

plt.xlabel("Age")
plt.title("Age Distribution within classes")
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')

ax5 = plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(df.Embarked.value_counts()))
plt.title("Passengers per boarding location")
plt.show()
