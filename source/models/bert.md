# BERT Model

自然言語処理モデル

## インストール

```
!pip install transformers fugashi ipadic
```

## 日本語のトークン, ID取得

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

## ID,トークンからもとの文章に変換

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



## BERTによるベクトル化

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



## 文書分類

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



