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
ds = ds.shuffle(100, seed=2).batch(10).repeat(2)
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

batchを作成する際に端数が出ないようにするには`drop_remainder`を指定する

```
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices(tf.range(20))
ds = ds.shuffle(10).batch(9, drop_remainder=True).repeat(2)
for data in ds:
    print(data)
---
tf.Tensor([ 7 10  3  9  2 12  5  4 14], shape=(9,), dtype=int32)
tf.Tensor([13 16  6 15  8 17 18 19  1], shape=(9,), dtype=int32)
tf.Tensor([ 6  1  8  3 13  2  0 12 11], shape=(9,), dtype=int32)
tf.Tensor([ 7 19 14 17  4 18  9 15 10], shape=(9,), dtype=int32)
```

## よく使われるdataset method

https://www.tensorflow.org/api_docs/python/tf/data/Dataset


### map

個々のデータに対して何らかの処理を行う

```
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
ds = ds.map(lambda x: x*2)
for data in ds:
  print(data)
---
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)
tf.Tensor(10, shape=(), dtype=int32)
tf.Tensor(12, shape=(), dtype=int32)
tf.Tensor(14, shape=(), dtype=int32)
tf.Tensor(16, shape=(), dtype=int32)
tf.Tensor(18, shape=(), dtype=int32)
```


### concatenate

結合する

```
a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
ds = a.concatenate(b)
list(ds.as_numpy_iterator())
---
[1, 2, 3, 4, 5, 6, 7]
```

### zip

pythonのzipと同じイメージ

```
a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
ds = tf.data.Dataset.zip((a, b))
list(ds.as_numpy_iterator())
---
[(1, 4), (2, 5), (3, 6)]
```

### window

datasetをサブセットで分ける

```
dataset = tf.data.Dataset.range(7).window(3)
for window in dataset:
  print(window)

for window in dataset:
  print([item.numpy() for item in window])
---
<_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
<_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
<_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
[0, 1, 2]
[3, 4, 5]
[6]
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


## CSVのパース

tf.io.decode_csvで欠損値などを補いながら読み込める。defsは欠損値に対するdefault値

```
line = "5,1,20.9,2,"
defs = [0.] * 4 + [tf.constant([99], dtype=tf.float32)]
fields = tf.io.decode_csv(line, record_defaults=defs)
fields
---
[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=20.9>,
 <tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=99.0>]
```

このままではただのListなので１次元テンソルに変換して一つのデータセットとする
```
tf.stack(fields)
---
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 5. ,  1. , 20.9,  2. , 99. ], dtype=float32)>
```

正解ラベルを含んでいる場合は以下のように分離する

```
x = tf.stack(fields[:-1])
y = tf.stack(fields[-1:])
print(x)
print(y)
---
tf.Tensor([ 5.   1.  20.9  2. ], shape=(4,), dtype=float32)
tf.Tensor([99.], shape=(1,), dtype=float32)
```

## Datasetリソース例

### ファイル

例: 
```
data01 = pd.read_csv("data/01.csv")
data02 = pd.read_csv("data/02.csv")
print(data01.head())
print(data02.head())
---
   Unnamed: 0  ID    name  age
0           0   1    Taro   10
1           1   2    Jiro    9
2           2   3  Saburo    8
   Unnamed: 0  ID  name  age
0           0   4  Siro    7
1           1   5  Goro    6
```

```
filepaths = ["data/01.csv", "data/02.csv"]
#filepaths = ["data/*.csv"]  <- こっちでも可
filepath_dataset = tf.data.Dataset.list_files(filepaths, seed=10)
for f in filepath_dataset:
  print(f)
---
tf.Tensor(b'data/01.csv', shape=(), dtype=string)
tf.Tensor(b'data/02.csv', shape=(), dtype=string)
```

複数のファイル(テキスト)をシーケンシャルに読むのではなく、インターリーブで読む場合、interleave()を使用する

```
n_readers = 2
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers
)
for data in dataset:
  print(data)
---
tf.Tensor(b'0,4,Siro,7', shape=(), dtype=string)
tf.Tensor(b'0,1,Taro,10', shape=(), dtype=string)
tf.Tensor(b'1,5,Goro,6', shape=(), dtype=string)
tf.Tensor(b'1,2,Jiro,9', shape=(), dtype=string)
tf.Tensor(b'2,3,Saburo,8', shape=(), dtype=string)
```

### Dataset Pipeline例

datasetにはprefetch(1)を用いることで先読みによる効率化が期待できる。1あれば基本は十分事足りる

```
def preprocess(csv_line):
  defs = [0.]*4
  fields = tf.io.decode_csv(csv_line, record_defaults=defs)
  x = tf.stack(fields[:-1])
  y = tf.stack(fields[-1:])
  return x, y

def csv_reader_dataset(filepaths, repeat=1, n_readers=5, 
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
  dataset = tf.data.Dataset.list_files(filepaths)
  dataset = dataset.interleave(
      lambda filepath: tf.data.TextLineDataset(filepath).skip(1), # Skip header
      cycle_length=n_readers, num_parallel_calls=n_parse_threads
  )
  dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
  dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
  return dataset.prefetch(1)
```
使い方
```
dataset = csv_reader_dataset(["data/*.csv"])
print(dataset)
for x, y in dataset:
  print(x.numpy(), y.numpy())
```

## 標準のデータセットを使う

tensorflow側で用意されているデータセットを使用する場合はtfdsを使用する

* 例: mnist

```
import tensorflow_datasets as tfds 
dataset = tfds.load(name="mnist", batch_size=32)

data = next(iter(dataset["train"]))
print(data.keys())
print(data["image"].shape)
print(data["label"].shape)
---
dict_keys(['image', 'label'])
(32, 28, 28, 1)
(32,)
```

as_supervised=Trueをつけるとtuplle形式で得られる。このほうがtensorflowで使用するえでは楽

```
import tensorflow_datasets as tfds 
dataset = tfds.load(name="mnist", batch_size=32, as_supervised=True)

data = next(iter(dataset["train"]))
print(data[0].shape)
print(data[1].shape)
---
(32, 28, 28, 1)
(32,)
```

* データセットにせずにそのままのデータを使う場合

```
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
```
