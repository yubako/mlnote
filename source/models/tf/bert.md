# BERT Model

自然言語処理モデル

## インストール

```
!pip install transformers fugashi ipadic
```

## 前処理関連

### 日本語のトークン, ID取得

```
from transformers import BertJapaneseTokenizer

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

texts = ["本日は晴天なり", "散歩すべし"]
encoding = tokenizer(
    texts, max_length=12, 
    truncation=True, 
    padding="max_length"
)
print(encoding)
```
```
{'input_ids':
    [[2, 108, 28486, 9, 4798, 28849, 297, 3, 0, 0, 0, 0], 
     [2, 19253, 340, 16513, 3, 0, 0, 0, 0, 0, 0, 0]], 
  'token_type_ids':
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
  'attention_mask':
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
}
```

### ID,トークンからもとの文章に変換

```
tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
print(tokens)

text = tokenizer.convert_tokens_to_string(tokens)
print(text)
```
```
['[CLS]', '本', '##日', 'は', '晴', '##天', 'なり', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
[CLS] 本日 は 晴天 なり [SEP] [PAD] [PAD] [PAD] [PAD]
```


### 正規化

全角、半角を統一する (NFKCは正規化のモード)

```
import unicodedata

normalize = lambda x: unicodedata.normalize("NFKC", x)
normalize("A　B　C")
```


### BERTによるベクトル化

Tensorflow用はTFBertModel, BertModelはpytorch用

```
from transformers import TFBertModel

texts = ["本日は晴天なり", "散歩すべし"]
encoding = tokenizer(
    texts, max_length=12, 
    truncation=True, 
    padding="max_length",
    return_tensors="tf"    # Use tensorflow object
)

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
bert = TFBertModel.from_pretrained(model_name)
print(bert.config)

output = bert(**encoding)
print(output.last_hidden_state.shape)
```
```
BertConfig {
  "_name_or_path": "cl-tohoku/bert-base-japanese-whole-word-masking",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "tokenizer_class": "BertJapaneseTokenizer",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 32000
}

(2, 32, 768)
```

### datasetの作成例

datasetのmapはグラフモードで動作するので、通常の関数を実行する場合は`th.py_function`でwrapする

```
def preprocess(text, target=-1):
  if tf.is_tensor(text):
    text = bytes.decode(text.numpy(), encoding="utf-8")
  tokens = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="tf")
  tokens = np.array([tokens["input_ids"], tokens["token_type_ids"], tokens["attention_mask"]], dtype=np.int32)
  tokens = tokens.squeeze(axis=1)
  return tf.constant(tokens), tf.cast(target, tf.int32)

def make_dataset(data):

  ds = tf.data.Dataset.from_tensor_slices((data["text"], data["target"]))
  ds = ds.map(lambda text, target: tf.py_function(preprocess, inp=[text, target], Tout=[tf.int32, tf.int32]))
  ds = ds.shuffle(100)
  ds = ds.batch(32).prefetch(1)
  return ds

ds_train = make_dataset(train)

```


## 文章の穴埋め

TFBertForMaskedLM

```
import numpy as np
from transformers import TFBertForMaskedLM, AutoTokenizer
import tensorflow as tf

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_mlm = TFBertForMaskedLM.from_pretrained(model_name)

text = "今日は[MASK]へ行く。"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['今日', 'は', '[MASK]', 'へ', '行く', '。']

input_ids = tokenizer.encode(text, return_tensors="tf")
output = bert_mlm(input_ids=input_ids)

# [MASK]位置はid値から特定
maskid = tokenizer.vocab["[MASK]"]
index = tf.where(input_ids == maskid)  # Return like [[0 3]]
r = tf.argmax(output.logits[0, index[0][1]], axis=-1)
mask = tokenizer.convert_ids_to_tokens([r])
print(mask)
print(text.replace("[MASK]", mask[0]))
---
['東京']
今日は東京へ行く。
```

## 文章分類

BertForSequenceClassification

2値分類を行う

```
import transformers
from transformers import AutoTokenizer, TFBertForSequenceClassification

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_sc = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

texts = [
         "この映画は面白かった",
         "この映画の最後にはがっかりさせられた",
         "この映画をみて幸せな気持ちになった"
]
label_list = [1, 0, 0]
encoding = tokenizer(texts, padding="longest", return_tensors="tf")
print(encoding.keys())
bert_sc.summary()
output = bert_sc(**encoding)
output
---
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
Model: "tf_bert_for_sequence_classification_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bert (TFBertMainLayer)      multiple                  110617344 
                                                                 
 dropout_75 (Dropout)        multiple                  0         
                                                                 
 classifier (Dense)          multiple                  1538      
                                                                 
=================================================================
Total params: 110,618,882
Trainable params: 110,618,882
Non-trainable params: 0
_________________________________________________________________
TFSequenceClassifierOutput([('logits',
                             <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
                             array([[ 0.3863358 , -0.05458079],
                                    [ 0.3874793 , -0.20349072],
                                    [ 0.10936061, -0.20661315]], dtype=float32)>)])
```

### サンプルコード

bertのモデルには`input_ids`, `attention_mask`, `token_type_ids`の3つが必要になるので
datasetの作成とモデルの作成にひと工夫ひつよう

livedoorの記事を使用
```
!pip install transformers fugashi ipadic tensorflow-gpu
!wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
!tar xfz ldcc-20140209.tar.gz
```

```
import tensorflow as tf
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertForSequenceClassification

# Tokenizer作成
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# token最大長
max_length = 128
category_list = [
  "dokujo-tsushin",
  "it-life-hack", 
  "kaden-channel",
  "livedoor-homme",
  "movie-enter"
  "peachy",
  "smax",
  "sports-watch",
  "topic-news"
]
num_labels=len(category_list)

# ファイルの読み込みと、tokenizerによるベクトル化
dataset = {"text": [], "label": []}
for label, category in enumerate(category_list):
  print(category)
  for file in glob.glob(f"text/{category}/{category}*"):
    lines = open(file).read().splitlines()
    text = "\n".join(lines[3:])
    dataset["text"].append(text)
    dataset["label"].append(label)

dataset_tokens = dataset["text"].map(
    lambda x: tokenizer(x, max_length=max_length, padding="max_length", truncation=True)
)

# 訓練用と検証用に分割
x_train, x_valid, y_train, y_valid = train_test_split(
    dataset_tokens, dataset["label"], 
    test_size=0.2, shuffle=True, stratify=dataset["label"])


# データ設置を作成
data = {
    "train": (
       tf.data.Dataset.from_tensor_slices((
        {
         "input_ids": [x["input_ids"] for x in x_train],
          "attention_mask": [x["attention_mask"] for x in x_train],
          "token_type_ids": [x["token_type_ids"] for x in x_train]
        },
        y_train
      ))
    ),
    "validation": (
       tf.data.Dataset.from_tensor_slices((
        {
         "input_ids": [x["input_ids"] for x in x_valid],
          "attention_mask": [x["attention_mask"] for x in x_valid],
          "token_type_ids": [x["token_type_ids"] for x in x_valid]
        },
        y_valid
      ))
    )
}
train_ds = data["train"].shuffle(1000).batch(32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = data["validation"].batch(32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# モデルの作成
def make_model(bert, bert_frozen=True):

  bert.trainable = True
  # input
  input_ids = tf.keras.layers.Input(shape=(max_length, ), dtype=tf.int32, name="input_ids")
  attention_mask  = tf.keras.layers.Input(shape=(max_length, ), dtype=tf.int32, name="attention_mask")
  token_type_ids = tf.keras.layers.Input(shape=(max_length, ), dtype=tf.int32, name="token_type_ids")
  inputs = [input_ids, attention_mask, token_type_ids]

  # bert
  x = bert.layers[0](inputs)

  # only use pooled_output
  out = x.pooler_output
  out = tf.keras.layers.Dropout(0.1)(out)
  out = tf.keras.layers.Dense(units=64, activation="relu")(out)
  out = tf.keras.layers.Dropout(0.1)(out)
  out = tf.keras.layers.Dense(units=num_labels, activation="softmax")(out)
  return tf.keras.Model(inputs=inputs, outputs=out)

bert = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model = make_model(bert)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())

tf.keras.utils.plot_model(model)
model.summary()
history = model.fit(train_ds, validation_data=valid_ds, epochs=10)
```



## 多値分類

BertModelでone-hot encodingのデータを使用して多値分類を行う

```
class BertForSequenceClassificationMultiLabel(tf.keras.Model):
  def __init__(self, num_labels, **kwargs):
    super().__init__(**kwargs)
    
    self.bert = TFBertModel.from_pretrained(model_name)

  def call(self, inputs):

    (input_ids, token_type_ids, attention_mask) = inputs
    x = self.bert(
          input_ids=input_ids,
          token_type_ids=token_type_ids,
          attention_mask=attention_mask
    )

    mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), dtype=tf.float32)
    sum = tf.math.reduce_sum(x.last_hidden_state * mask, axis=1)
    return x


def make_model():

  inputs = [
    tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids"),
    tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids"),
    tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
  ]
  bert = BertForSequenceClassificationMultiLabel(num_labels=3)
  out = bert(inputs)
  out = tf.keras.layers.Dense(units=3, activation=None)(out)
  return tf.keras.Model(inputs=inputs, outputs=out)

model = make_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.BinaryAccuracy())
---
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_ids (InputLayer)         [(None, 128)]        0           []                               
                                                                                                  
 token_type_ids (InputLayer)    [(None, 128)]        0           []                               
                                                                                                  
 attention_mask (InputLayer)    [(None, 128)]        0           []                               
                                                                                                  
 bert_for_sequence_classificati  (None, 768)         110617344   ['input_ids[0][0]',              
 on_multi_label_23 (BertForSequ                                   'token_type_ids[0][0]',         
 enceClassificationMultiLabel)                                    'attention_mask[0][0]']         
                                                                                                  
 dense_35 (Dense)               (None, 3)            2307        ['bert_for_sequence_classificatio
                                                                 n_multi_label_23[0][0]']         
                                                                                                  
==================================================================================================
Total params: 110,619,651
Trainable params: 110,619,651
Non-trainable params: 0
```

### 形態素解析

文章を形態素に分割して品詞などを判別する

例文: しょうゆを１本買いました

```
from sudachipy import tokenizer
from sudachipy import dictionary

text = "しょうゆを１本買いました"

t = dictionary.Dictionary(dict="core").create()
t.tokenize(text)
---
<MorphemeList[
  <Morpheme(しょうゆ, 0:4, (0, 72322))>,
  <Morpheme(を, 4:5, (0, 171705))>,
  <Morpheme(１, 5:6, (0, 17))>,
  <Morpheme(本, 6:7, (0, 511818))>,
  <Morpheme(買い, 7:9, (0, 692222))>,
  <Morpheme(まし, 9:11, (0, 148938))>,
  <Morpheme(た, 11:12, (0, 83861))>,
]>
```

各形態素の参照方法

```
text = "しょうゆを１本買いました"

t = dictionary.Dictionary(dict="core").create()
mode = tokenizer.Tokenizer.SplitMode.C

tokens = t.tokenize(text, mode=mode)
print(tokens[0].surface())             # 表層形(もとの文字列表現)
print(tokens[0].normalized_form())     # 正規化した表記
print(tokens[0].reading_form())        # ふりがな(読みのカタカナ表記)
print(tokens[0].begin())               # 開始位置
print(tokens[0].end())                 # 終了位置
---
しょうゆ
醤油
ショウユ
0
4
```

sudachiの正規化は以下のようになる

```
text = "しょうゆを１本買いました"

t = dictionary.Dictionary(dict="core").create()
mode = tokenizer.Tokenizer.SplitMode.C

for t in t.tokenize(text, mode=mode):
  origin = t
  normalized = t.normalized_form()
  print(origin," -> ",  normalized)

---
しょうゆ  ->  醤油
を  ->  を
１  ->  1
本  ->  本
買い  ->  買う
まし  ->  ます
た  ->  た
```

SplitModeの違いは以下

https://github.com/WorksApplications/SudachiPy

```
mode = tokenizer.Tokenizer.SplitMode.C
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家公務員']

mode = tokenizer.Tokenizer.SplitMode.B
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家', '公務員']

mode = tokenizer.Tokenizer.SplitMode.A
[m.surface() for m in tokenizer_obj.tokenize("国家公務員", mode)]
# => ['国家', '公務', '員']
```



## 固有表現抽出

文章から固有値を抽出する

固有値の表現方法

例文: AさんはBCD株式会社を起業した
```
entities = [
  {"name": "A", "span": [0, 1], "type": "人名", "type_id": 1},
  {"name": "BCD株式会社", "span": [4, 11], "type": "組織名", "type_id": 2}
]
```
* name: 固有表現のテキスト
* span: 固有表現の出現位置
* type: 固有表現種別
* type_id: 固有表現の種別識別子


学習過程

文字列を単語単位にトークン分割し、トークンに対して固有表現に相当するtype_idを推論して学習を行う。

例文: AさんはBCD株式会社を起業した

1. 単語単位で分割(形態素解析)

    ```
    tokens   = ["A", "さん", "は", "BCD", "株式", "会社", "を", "起業", "し", "た"]
    type_ids = [1,    0,     0,    2,     2,       2,     0,   0,      0,    0]
    ```

1. tokensをBERT入力用のデータとしてinput_ids, attention_mask, token_type_idsに符号化して、type_idsを正解ラベルとして学習する

Sample Code

```
from sudachipy import tokenizer
from sudachipy import dictionary
from transformers import AutoTokenizer, TFBertForTokenClassification
import tensorflow as tf
import numpy as np

max_length = 128

class Tokenizer():

  def __init__(self):
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    self.tokenizer_obj = dictionary.Dictionary(dict="core").create()
    self.tokenizer_mode = tokenizer.Tokenizer.SplitMode.A
    self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  def tokenize(self, text):
    morphemes = self.tokenizer_obj.tokenize(text, mode=self.tokenizer_mode)
    tokens = []
    for m in morphemes:
      tokens.append(m.surface())
    return tokens

  def tokenize_with_labels(self, text, entities):
    morphemes = self.tokenize(text)
    labels = []
    tokens = []
    pos = 0

    entities = sorted(entities, key=lambda x: x["span"][0])
    for entity in entities:
      start = entity["span"][0]
      end = entity["span"][1]
      token = text[pos:start]
      entity_token = text[start:end]
      
      txt_tokens = self.tokenize(token)
      txt_labels = [0] * len(txt_tokens)

      ent_tokens = self.tokenize(entity_token)
      ent_labels = [entity["type_id"]] * len(ent_tokens)

      tokens.extend(txt_tokens)
      tokens.extend(ent_tokens)

      labels.extend(txt_labels)
      labels.extend(ent_labels)
      pos = end

    if len(entities) > 0:
      txt_tokens = self.tokenize(text[entities[-1]["span"][1]:])
      txt_labels = [0] * len(txt_tokens)
      tokens.extend(txt_tokens)
      labels.extend(txt_labels)
    return tokens, labels

  def encode_token_labels(self, tokens, labels, max_length=128):
    # CLS, SEPが入るので-2
    if len(tokens) > (max_length-2):
      tokens = tokens[:max_length-2]
      labels = labels[:max_length-2]

    input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
    encodes = bert_tokenizer.prepare_for_model(
        input_ids,
        padding="max_length", 
        max_length=max_length, 
        trucation=True,
        return_tensors="tf"
    )
    if len(encodes["input_ids"]) > max_length:
      print("truncated: ", len(encodes["input_ids"]))
      encodes["input_ids"] = tf.ragged.constant(encodes["input_ids"][:max_length])
      encodes["attention_mask"] = tf.ragged.constant(encodes["attention_mask"][:max_length])
      encodes["token_type_ids"] = tf.ragged.constant(encodes["token_type_ids"][:max_length])

    # [CLS] [SEP] [PAD]のラベルは0に設定する
    labels = [0] + labels
    labels = labels + [0] * (max_length - len(labels))
    return encodes, labels

```

* 単語単位で分割
```
text = "AさんはBCD株式会社を起業しました。"
obj = Tokenizer()
print(obj.tokenize(text))
---
['A', 'さん', 'は', 'BCD', '株式', '会社', 'を', '起業', 'し', 'まし', 'た', '。']
```

* Entityをもとにlabelを作成
```
type_ids = {
    1: "人物",
    2: "企業名"
}
entities = [
  {"name": "A", "span": [0, 1], "type_id": 1},
  {"name": "BCD株式会社", "span": [4, 11], "type_id": 2}
]
obj = Tokenizer()
tokens, labels = obj.tokenize_with_labels(text, entities)
print(tokens)
print(labels)
---
['A', 'さん', 'は', 'BCD', '株式', '会社', 'を', '起業', 'し', 'まし', 'た', '。']
[1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0]
```

* 生成したtokenとlabelをbert入力形式に変換する

```
encoded_tokens, encoded_labels = obj.encode_token_labels(tokens, labels)
print(encoded_tokens.keys())
print(encoded_labels[:10])
---
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
[0, 1, 0, 0, 2, 2, 2, 0, 0, 0]
```



