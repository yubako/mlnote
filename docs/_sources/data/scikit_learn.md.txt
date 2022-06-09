# scikit-learn


## スケール操作

* 標準化 (StandardScaler)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

平均が0, 分散が1 にする。すべての特徴量の大きさを抑えてくれるが、一定の範囲に収まることを保証するわけではない


* RobustScaler

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

個々の特徴量の大きさを抑えてくれるが、平均値と分散の代わりに中央値と四分位数(*)を用いる。  
(*) 四分位数について
```
  中央値x -> 集合の半分がxよりも大きく、もう半分がxより小さい
  第一四分位数  -> 全体の1/4がxより小さく、3/4がxよりも大きいような数
  第三四分位数  -> 全体の3/4がxより小さく、1/4がxよりも大きいような数
```

RobustScalerは極端に他の値と異なるような値(ハズレ値)を無視する。


* MinMaxScaler

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

データがちょうど0〜1の間に入るように変換する


* Pandas DataFrameに対するスケール操作

```
imr.fit(X_train['Age'].values.reshape(-1, 1))
X_train['Age'] = imr.transform(X_train['Age'].values.reshape(-1, 1))
X_valid['Age'] = imr.transform(X_valid['Age'].values.reshape(-1, 1))

std.fit(X_train['Age'].values.reshape(-1, 1))
X_train['Age'] = std.transform(X_train['Age'].values.reshape(-1, 1))
X_valid['Age'] = std.transform(X_valid['Age'].values.reshape(-1, 1))
```


## 訓練データと検証データ分割

```
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
```