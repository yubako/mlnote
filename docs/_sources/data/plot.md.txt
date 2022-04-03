# Plot

## 散布図行列(ペアプロット)

pandasを用いてplotできる(trainはDataFrame)

[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html)

```
from sklearn import datasets

d = datasets.load_iris()
iris = pd.DataFrame(d.data)
grr = pd.plotting.scatter_matrix(iris, figsize=(15, 15), range_padding=0.5, c=d.target)
```

matplotlibでごりごり書く場合
```
columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
survived = train['Survived']

fig = plt.figure(figsize=(15, 15))
for i, col1 in enumerate(columns):
    for j, col2 in enumerate(columns):
        ax = fig.add_subplot(5, 5, 5*i+j+1)
        x = train[col1]
        y = train[col2]
        ax.set_title("%s - %s" % (col1, col2))
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.scatter(x, y, c=survived, marker="o")

plt.tight_layout()
plt.show()
```
