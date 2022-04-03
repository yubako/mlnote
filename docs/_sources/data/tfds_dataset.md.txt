# Tensorflow dataset

## Datasetの作成

```
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices([1,2,3])
dataset = tf.data.Dataset.from_tensor_slices(([1,2,3], [4, 5, 6]))
```

## batch, repeat, shuffle

```
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices(tf.range(100))
ds = ds.shuffle(100).batch(10).repeat(2)
for data in ds:
    print(data)
---
tf.Tensor([15 16 35 81 25 30 73 74 88  4], shape=(10,), dtype=int32)
tf.Tensor([67 90 89 60 28 87 22 34 56 11], shape=(10,), dtype=int32)
tf.Tensor([ 5 23 77 52 20 91 47 93 66 50], shape=(10,), dtype=int32)
tf.Tensor([19 98 62 17 42 13 78 80 27 51], shape=(10,), dtype=int32)
tf.Tensor([92 94 70 43 96 97 46 95  9 36], shape=(10,), dtype=int32)
tf.Tensor([33 32 85 41 48 29 44 84 57 45], shape=(10,), dtype=int32)
tf.Tensor([54 99 71 86 21 65 72  0 49 10], shape=(10,), dtype=int32)
tf.Tensor([59 68  6 75 63 55 39 83 40 12], shape=(10,), dtype=int32)
tf.Tensor([ 3 69 18 37 14  7 38 76  8  1], shape=(10,), dtype=int32)
tf.Tensor([24 26 58  2 31 61 64 79 82 53], shape=(10,), dtype=int32)
tf.Tensor([64 16 84 80 53 17 23  6 48 86], shape=(10,), dtype=int32)
tf.Tensor([13 21 70 89 54 19 91 31 30 79], shape=(10,), dtype=int32)
tf.Tensor([69 68 29 87 50 97 98 40 95 62], shape=(10,), dtype=int32)
tf.Tensor([12  0 74 92 34 42 41 27  1 81], shape=(10,), dtype=int32)
tf.Tensor([93 75 52 11 63 22  2 58 73 49], shape=(10,), dtype=int32)
tf.Tensor([46 66 35 43 36 18 72 90 20 24], shape=(10,), dtype=int32)
tf.Tensor([ 8 51 38 25 99 44  4 32 61 67], shape=(10,), dtype=int32)
tf.Tensor([10 26 28  3 57 94 96 56 47 77], shape=(10,), dtype=int32)
tf.Tensor([82  9 55 59 45 60 15 33  7  5], shape=(10,), dtype=int32)
tf.Tensor([76 14 78 85 88 83 39 71 65 37], shape=(10,), dtype=int32)

```

## データセットの数を取得する

訓練時のepochごとのstep数を計算するのに使う

```
dataset = tf.data.Dataset.from_tensor_slices(([1,2,3], [4, 5, 6]))
steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
print(steps_per_epoch)

dataset = dataset.repeat(10)
steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
print(steps_per_epoch)
---
3
30
```