# 前処理レイヤー

データに対する事前処理をlayerで行う

https://www.tensorflow.org/guide/keras/preprocessing_layers?hl=ja

## TextVectorization

例: テキストのベクトル変換を行う。

```
from tensorflow.keras.layers.experimental import preprocessing

def make_model():
  text = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="text")
  lvec = preprocessing.TextVectorization(
      output_sequence_length=64,
  )
  lvec.adapt(train["text"])

  vecs = lvec(text)
  x = vecs
  vocab = lvec.get_vocabulary()
  x = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=64)(x)
  x = tf.keras.layers.LSTM(64, return_sequences=True, activation="tanh")(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.LSTM(64, return_sequences=True, activation="tanh")(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.LSTM(2)(x)
  return tf.keras.Model(inputs=text, outputs=x)

model = make_model()
model.summary()
```
