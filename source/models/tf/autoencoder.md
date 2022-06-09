# オートエンコーダ

潜在表現、特徴量学習に使う

* fashion_mnistを使った例

28x28のグレースケール画像の特徴を学習し、サイズ30の配列で表現する

```
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

stacked_encoder = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation="selu"),
  tf.keras.layers.Dense(30, activation="selu"),
])
stacked_decoder = tf.keras.models.Sequential([
  tf.keras.layers.Dense(100, input_shape=(30,)),
  tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
  tf.keras.layers.Reshape((28, 28))
])

stacked_ae = tf.keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                   optimizer=tf.keras.optimizers.SGD(lr=1.5))
stacked_ae.summary()

stacked_ae.fit(X_train, X_train, epochs=10, validation_data=(X_valid, X_valid))
```

学習した結果は以下で確認できる

```
import matplotlib.pyplot as plt
def plot_image(image):
  plt.imshow(image, cmap="binary")
  plt.axis("off")

def show_reconstructions(model, n_images=5):
  reconstructions = model.predict(X_valid[:n_images])
  fig = plt.figure(figsize=(n_images * 1.5, 3))
  for image_index in range(n_images):
    plt.subplot(2, n_images, 1 + image_index)
    plot_image(X_valid[image_index])
    plt.subplot(2, n_images, 1 + n_images + image_index)
    plot_image(reconstructions[image_index])

show_reconstructions(stacked_ae)
```

また28x28から30に次元削減ができているため、更にt-SNEを用いて次元削減することで可視化することもできる。  
オートエンコーダによる次元削減だけでは可視化するのは難しいが、特徴量が多い大規模データセットをあつかう
ことができるため、オートエンコーダによる次元削減＋その他の次元削減アルゴリズムで可視化、といったことができる。

```
from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2d = tsne.fit_transform(X_valid_compressed)
plt.scatter(X_valid_2d[:, 0], X_valid_2d[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()
```

特徴量を学習できるため、事前にオートエンコーダで特徴量を学習したあと、ラベル付きデータで学習すれば、
すべての学習データにラベル付けをすることなく学習することができる。  

```
Phase-1
          [エンコーダ]      [デコーダ]
  入力 --> 隠れ層1 --> 隠れ層2 --> 隠れ層3 --> 出力
            |           |
            | 重みcopy   | 重みcopy
Phase-2     v           v
  入力 --> 隠れ層1 --> 隠れ層2 --> 全結合層 --> SoftMax層
```

CNNでオートエンコーダを作る場合は以下のように作る

```
conv_encoder = tf.keras.models.Sequential([
  tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
  tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu"),
  tf.keras.layers.MaxPool2D(pool_size=2),
  tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu"),
  tf.keras.layers.MaxPool2D(pool_size=2),
  tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"),
  tf.keras.layers.MaxPool2D(pool_size=2)
])

conv_decoder = tf.keras.models.Sequential([
  tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="valid", activation="selu", input_shape=(3, 3, 64)),
  tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu"),
  tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
  tf.keras.layers.Reshape([28, 28])
])

conv_ae = tf.keras.models.Sequential([conv_encoder, conv_decoder])
```

# 変分オートエンコーダ(VAE)

変分オートエンコーダは生成的オートエンコーダで、訓練セットからサンプリングされたかのように見える
新しいインスタンスを作ることができる


```
import tensorflow as tf 
class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    mean, log_var = inputs
    return tf.keras.backend.random_normal(tf.shape(log_var)) * tf.keras.backend.exp(log_var / 2) + mean

codings_size = 10

# Encoder Model
inputs = tf.keras.layers.Input(shape=(28, 28))
z = tf.keras.layers.Flatten()(inputs)
z = tf.keras.layers.Dense(150, activation="selu")(z)
z = tf.keras.layers.Dense(100, activation="selu")(z)
codings_means = tf.keras.layers.Dense(codings_size)(z) # μ
codings_log_var = tf.keras.layers.Dense(codings_size)(z) # γ
codings = Sampling()([codings_means, codings_log_var])
variational_encoder = tf.keras.Model(inputs=[inputs], outputs=[codings_means, codings_log_var, codings])

# Decoder Model
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="selu")(x)
x = tf.keras.layers.Dense(28*28, activation="sigmoid")(x)
output = tf.keras.layers.Reshape((28, 28))(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[output])

# 変分オートエンコーダModel
_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

# 潜在損失関数を追加してCompile
latent_loss = -0.5 * tf.keras.backend.sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_means),
    axis=-1
)
variational_ae.add_loss(tf.reduce_mean(latent_loss) / 784.)
variational_ae.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer="rmsprop")
history = variational_ae.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_valid, X_valid))

# 任意のコーディング層の値を用いてインスタンスを生成
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings)

fig = plt.figure(figsize=(8, 6))
for i in range(len(images)):
  ax = fig.add_subplot(3, 4, i+1)
  ax.imshow(images[i])
plt.show()
```

