# Numpy

## numpyとtf.Tensor変換

```
import numpy as np
import tensorflow as tf

# numpy配列作成
np_a = np.array(range(10), dtype=np.int16)
print(np_a)

# numpy -> tensorflow.Tensor
tf_a = tf.convert_to_tensor(np_a)
print(tf_a)

# Tensor -> numpy
np_a = tf_a.numpy()
print(np_a)
```

## sortのaxis挙動

* 基本データ

```
import tensorflow as tf
a = tf.convert_to_tensor(np.array([
        [[2, 4, 9, 14],[0, 11, 17, 4], [23, 2, 9, 12]],
        [[1, 2, 3, 4],[8, 3, 14, 5], [6, 7, 8, 9]]
      ], dtype=np.int8)
)
print("--- Data ---\n", a)
print("--- Shape ---\n", a.shape)
---
--- Data ---
 tf.Tensor(
[[[ 2  4  9 14]
  [ 0 11 17  4]
  [23  2  9 12]]

 [[ 1  2  3  4]
  [ 8  3 14  5]
  [ 6  7  8  9]]], shape=(2, 3, 4), dtype=int8)
--- Shape ---
 (2, 3, 4)
```

* axis=0

`a[:, 0, 0]`の単位でソートされる

```
for i in range(a.shape[1]):
  for j in range(a.shape[2]):
    print(a[:, i, j])
---
tf.Tensor([2 1], shape=(2,), dtype=int8)
tf.Tensor([4 2], shape=(2,), dtype=int8)
tf.Tensor([9 3], shape=(2,), dtype=int8)
tf.Tensor([14  4], shape=(2,), dtype=int8)
tf.Tensor([0 8], shape=(2,), dtype=int8)
tf.Tensor([11  3], shape=(2,), dtype=int8)
tf.Tensor([17 14], shape=(2,), dtype=int8)
tf.Tensor([4 5], shape=(2,), dtype=int8)
tf.Tensor([23  6], shape=(2,), dtype=int8)
tf.Tensor([2 7], shape=(2,), dtype=int8)
tf.Tensor([9 8], shape=(2,), dtype=int8)
tf.Tensor([12  9], shape=(2,), dtype=int8)
```

```
print(a)
print(tf.sort(a, axis=0))
---
tf.Tensor(
[[[ 2  4  9 14]
  [ 0 11 17  4]
  [23  2  9 12]]

 [[ 1  2  3  4]
  [ 8  3 14  5]
  [ 6  7  8  9]]], shape=(2, 3, 4), dtype=int8)
tf.Tensor(
[[[ 1  2  3  4]
  [ 0  3 14  4]
  [ 6  2  8  9]]

 [[ 2  4  9 14]
  [ 8 11 17  5]
  [23  7  9 12]]], shape=(2, 3, 4), dtype=int8)
```

* axis=1

`a[0, :, 0]`の単位でソートされる

```
for i in range(a.shape[0]):
  for j in range(a.shape[2]):
    print(a[i, :, j])
---
tf.Tensor([ 2  0 23], shape=(3,), dtype=int8)
tf.Tensor([ 2  0 23], shape=(3,), dtype=int8)
tf.Tensor([ 4 11  2], shape=(3,), dtype=int8)
tf.Tensor([ 9 17  9], shape=(3,), dtype=int8)
tf.Tensor([14  4 12], shape=(3,), dtype=int8)
tf.Tensor([1 8 6], shape=(3,), dtype=int8)
tf.Tensor([2 3 7], shape=(3,), dtype=int8)
tf.Tensor([ 3 14  8], shape=(3,), dtype=int8)
tf.Tensor([4 5 9], shape=(3,), dtype=int8)
```

```
print(a)
print(tf.sort(a, axis=1))
---
tf.Tensor(
[[[ 2  4  9 14]
  [ 0 11 17  4]
  [23  2  9 12]]

 [[ 1  2  3  4]
  [ 8  3 14  5]
  [ 6  7  8  9]]], shape=(2, 3, 4), dtype=int8)
tf.Tensor(
[[[ 0  2  9  4]
  [ 2  4  9 12]
  [23 11 17 14]]

 [[ 1  2  3  4]
  [ 6  3  8  5]
  [ 8  7 14  9]]], shape=(2, 3, 4), dtype=int8)
```


## max, argmax

* max 最大値を取得する
* argmax 最大値を持つindexを取得する


サンプルデータ
```
import numpy as np
a = np.random.rand(12).reshape(2, 2, 3)
print(a)
---
[[[0.1546972  0.36631914 0.72881664]
  [0.38821544 0.92575652 0.42821867]]

 [[0.40778945 0.98278608 0.97056111]
  [0.49367194 0.99666699 0.44673195]]]
```

* axis省略

```
print(a.max())
print(a.argmax())
---
0.9966669932873805
10
```

* axis=0
```
print(a.max(axis=0))
print(a.argmax(axis=0))
for i in range(a.shape[1]):
  for j in range(a.shape[2]):
    print(a[:, i, j], "->", np.max(a[:, i, j]))
---
[[0.40778945 0.98278608 0.97056111]
 [0.49367194 0.99666699 0.44673195]]
[[1 1 1]
 [1 1 1]]
[0.1546972  0.40778945] -> 0.4077894503642825
[0.36631914 0.98278608] -> 0.9827860815219699
[0.72881664 0.97056111] -> 0.9705611115370159
[0.38821544 0.49367194] -> 0.493671940776198
[0.92575652 0.99666699] -> 0.9966669932873805
[0.42821867 0.44673195] -> 0.4467319497523642
```

* axis=1

```
print(a.max(axis=1))
print(a.argmax(axis=1))
for i in range(a.shape[0]):
  for j in range(a.shape[2]):
    print(a[i, :, j], "->", np.max(a[i, :, j]))
---
[[0.38821544 0.92575652 0.72881664]
 [0.49367194 0.99666699 0.97056111]]
[[1 1 0]
 [1 1 0]]
[0.1546972  0.38821544] -> 0.38821544375069106
[0.36631914 0.92575652] -> 0.9257565230999869
[0.72881664 0.42821867] -> 0.7288166409375589
[0.40778945 0.49367194] -> 0.493671940776198
[0.98278608 0.99666699] -> 0.9966669932873805
[0.97056111 0.44673195] -> 0.9705611115370159
```

* axis=2

```
print(a.max(axis=2))
print(a.argmax(axis=2))
for i in range(a.shape[0]):
  for j in range(a.shape[1]):
    print(a[i, j, :], "->", np.max(a[i, j, :]))
---
[[0.72881664 0.92575652]
 [0.98278608 0.99666699]]
[[2 1]
 [1 1]]
[0.1546972  0.36631914 0.72881664] -> 0.7288166409375589
[0.38821544 0.92575652 0.42821867] -> 0.9257565230999869
[0.40778945 0.98278608 0.97056111] -> 0.9827860815219699
[0.49367194 0.99666699 0.44673195] -> 0.9966669932873805
```

