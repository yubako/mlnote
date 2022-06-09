# pandas


## CSVファイルの読み込み

```
import pandas as pd
train = pd.read_csv("/kaggle/input/titanic/train.csv")
```

## DataFrameの概要を表示する
```
train.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

## 全体統計情報を取得する

```
full.describe()
full.describe("O")
```

## DataFrameの連結

テストデータと訓練データを結合して全体の統計を見たいときなど

```
full = pd.concat([train, test], axis=0, sort=False)
```

## 欠損値の確認方法

```
train.isnull()
train.isnull().sum()
```

### 欠損値を削除する

行全体を削除する場合
```
train.dropa()
```

欠損値を含む列を削除する
```
train.dropa(axis=1)
```

### 欠損値を補完する

sklearnのSimpleImputeを使用する

[https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)



## データのプロファイリングを表示する

```
import pandas_profiling as pdp
pdp.ProfileReport(train)
```

## 列の削除

```
train = df.drop(["Survived"], axis=1)
labels = df["Survived"]
```