# Tensorflow

## 分類問題における損失関数

| 損失関数 | 用途 | 例(Logit) |
|---|---|---|
| BinaryCrossentropy | 二値分類 | y_true: 1 <br>y_pred: 0.69 |
| CategoricalCrossentropy | 多クラス分類 | y_true: [0][0][1]<br>y_pred: [0.30][0.40][0.55] |
| SparseCategoricalCrossentropy | 多クラス分類 | y_true: 2<br>y_pred: [0.30][0.15]0[0.55] |




