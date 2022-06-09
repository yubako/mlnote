# Preprocessing

## クラスラベルのエンコーディング

データ例
```
df = pd.DataFrame([
    ["green", "M", 10.1],
    ["red", "L", 13.5],
    ["blue", "XL", 15.3],
])
df.columns = ["color", "size", "price"]
```

### One-Hot エンコーディング

```
df_encoded = pd.get_dummies(df, columns=["color"])
---
  size  price  color_blue  color_green  color_red
0    M   10.1           0            1          0
1    L   13.5           0            0          1
2   XL   15.3           1            0          0
```

### マッピング

```
size_mapping = {"M": 1, "L": 2, "XL": 3}
df["size"] = df["size"].map(size_mapping)
---
   color  size  price
0  green     1   10.1
1    red     2   13.5
2   blue     3   15.3
```

## 前処理Layerの作成

### 標準化を行う事前処理Layer

```
import tensorflow as tf
import numpy as np 

class Standarzation(tf.keras.layers.Layer):
  def adapt(self, data_sample):
    self.means_ = np.mean(data_sample, axis=0, keepdims=True)
    self.stds_ = np.std(data_sample, axis=0, keepdims=True)
  
  def call(self, inputs):
    return (inputs - self.means_) / (self.stds_ + tf.keras.backend.epsilon())
```
※ spsilonは0除算を避けるための小さな値

事前にデータをいくつか与えて平均と標準偏差を求める必要がある

```
data_sample = np.ones(shape=(1, 100))
std_layer = Standarzation()
std_layer.adapt(data_sample)

model = tf.keras.Sequential()
model.add(std_layer)
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```


### ラベル特徴量を用いる場合

One Hot Encodingと埋め込み(Embedding)

* One Hot Encodingはカテゴリ種別ごとに列を追加して1 or 0で表現する
* 埋め込みはカテゴリを固定次元のベクトルで表現する

目安としてカテゴリ数が10個未満ならOne Hot Encodingがよさげ。50以上なら埋め込みのほうが良い。
10〜50の間はどっちともいえないので、両方つかってみて性能のいい方を選ぶ

**OneHotEncoding**

カテゴリ値の辞書を作成する

```
vocab = ["<1h OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```

oovはOut of Vocabularyの略。事前指定のカテゴリ外の値が入ったときにIDを割り振るためのバッファ  
以下のように使う
```
cat = tf.constant(["INLAND", "OTHERS"])
table.lookup(cat)
---
<tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 5])>
```

得られたindex値を用いてone hot encoding形式にする
```
cat = tf.constant(["INLAND", "OTHERS"])
indices = table.lookup(cat)
print(indices)
onehot = tf.one_hot(indices, depth=len(vocab) + num_oov_buckets)
print(onehot)
```

これをlayerと実装する場合は以下のようにする。tensor内はすべてデータ型は同じでないといけないので、
カテゴリデータと数値データはそれぞれ別のINPUTとして扱う

```
class OneHotEncode(tf.keras.layers.Layer):
  def __init__(self, num_oov_buckets=2, **kwargs):
    super().__init__(**kwargs)
    self.num_oov_buckets = num_oov_buckets

  def adapt(self, data_sample):
    cates, idx = tf.unique(data_sample)
    indices = tf.range(len(cates), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(cates, indices)
    self.table_ = tf.lookup.StaticVocabularyTable(table_init, self.num_oov_buckets)
    self.num_vocabs_ = len(cates) + self.num_oov_buckets
  
  def call(self, inputs):
    indices = self.table_.lookup(inputs)
    onehot = tf.one_hot(indices, depth=self.num_vocabs_)
    return onehot

sample_cate_data = np.array(["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BY", "INLAND"])
sample_regular_data = np.array([[1, 2], [2, 2], [3, 2], [1., 3], [9.1, 2]], dtype=np.float)

# Inputs
regular_inputs = tf.keras.Input(shape=[2])
category_inputs = tf.keras.Input(shape=[], dtype=tf.string)

# One Hot Encoding
onehot_encoder = OneHotEncode()
onehot_encoder.adapt(sample_cate_data)
onehot = onehot_encoder(category_inputs)

# 数値特徴量とOneHotEncoding済みの特徴量をつなげてmodelを作成する
encoded_inputs = tf.keras.layers.concatenate([regular_inputs, onehot], axis=-1)
model = tf.keras.models.Model(inputs=[regular_inputs, category_inputs], outputs=[encoded_inputs])

out = model((sample_regular_data, sample_cate_data))
out
---
<tf.Tensor: shape=(5, 8), dtype=float32, numpy=
array([[1. , 2. , 1. , 0. , 0. , 0. , 0. , 0. ],
       [2. , 2. , 0. , 1. , 0. , 0. , 0. , 0. ],
       [3. , 2. , 0. , 0. , 1. , 0. , 0. , 0. ],
       [1. , 3. , 0. , 0. , 0. , 1. , 0. , 0. ],
       [9.1, 2. , 1. , 0. , 0. , 0. , 0. , 0. ]], dtype=float32)>
```

**Embedding**

kerasにはEmbedding Layerがすでに存在するのでそれを使う

※ Embedding層は訓練可能な変数をもつためモデルに組み込むと重みが更新される。これを抑止する場合は
  Model外の事前処理として埋め込みを行うか、trainableをFasleにする


* Indexを引くためにLookup Table作成するのはOne Hot Encodingと同じ
```
vocab = ["<1h OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```

* Embedding layerを作成する。output_dimは出力ベクトルの次元を指定する

```
embedding_dim = 4
embedding = tf.keras.layers.Embedding(input_dim=len(vocab) + num_oov_buckets, output_dim=embedding_dim)
```

* 実行例

```
cates = tf.constant(["INLAND", "NEAR BAY"], dtype=tf.string)
indices = table.lookup(cates)
embedding(indices)
---
<tf.Tensor: shape=(2, 4), dtype=float32, numpy=
array([[-0.04014117, -0.04341874,  0.00825857,  0.01782556],
       [ 0.00501693, -0.00398098, -0.02564949, -0.02547029]],
      dtype=float32)>
```
