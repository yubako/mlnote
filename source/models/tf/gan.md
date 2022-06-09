# 敵対的生成ネットワーク

生成器(Generator)と判別器(Discriminator)をそれぞれ訓練して判別器が本物と間違えるようにする

訓練過程

1. 判別器を訓練する(生成器の重みは更新しない)
    本物は本物と、生成器が作ったものは偽物と判別できるように訓練する
    INPUT:
        * 訓練セットの本物の画像(ラベル: 1(本物))
        * 生成器が作成した贋作(ラベル: 0(偽物))

1. 生成器を訓練する(判別器の重みは更新しない)
    判別器が本物と判断するように訓練する
    INPUT:
        * 生成器が作成した贋作(ラベル: 1(本物)


```
codings_size = 30

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
  tf.keras.layers.Dense(150, activation="selu"),
  tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
  tf.keras.layers.Reshape((28, 28))
])
discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(150, activation="selu"),
  tf.keras.layers.Dense(100, activation="selu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])
gan = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.RMSprop())
# 生成域を訓練するとき(ganでtrain_on_batchを実行するとき)は重みは更新しないのでdiscriminatorの
# trainableはFalseにする
discriminator.trainable = False
gan.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.RMSprop())
```

通常の訓練ループでは処理できないので、独自ループを作る

```
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, coding_size, n_epochs=50):
  generator, discriminator = gan.layers
  for epoch in range(n_epochs):
    for X_batch in dataset:
      X_batch = tf.cast(X_batch, dtype=tf.float32)

      # 判別器の訓練
      noise = tf.random.normal(shape=(batch_size, coding_size), dtype=tf.float32)
      generated_images = generator(noise)
      X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
      y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
      discriminator.trainable = True
      discriminator.train_on_batch(X_fake_and_real, y1)

      # 生成器の訓練
      noise = tf.random.normal(shape=(batch_size, coding_size))
      y2 = tf.constant([[1.]] * batch_size)
      discriminator.trainable = False
      l = gan.train_on_batch(noise, y2)
    print(epoch, l)      
train_gan(gan, dataset, batch_size, codings_size)
```

お試し。あまり性能はよくない

```
noise = tf.random.normal(shape=(6, codings_size))
images = generator(noise)
fig = plt.figure(figsize=(8, 6))
for i, image in enumerate(images):
  ax = fig.add_subplot(2, 3, i+1)
  ax.imshow(image)
plt.show()
```


# 深層畳み込みGAN

CNNを用いたGAN.主な指針に以下のようなものがある

* プーリングそうを、判別器ではストライド付きの畳み込み層に。生成器では天地畳み込み層に置き換える
* 生成器の出力層と判別器の入力層以外はバッチ正規化を行う
* 全結合隠れ層を取り除く
* 生成器では出力層以外ですべてReLUを使う。出力層ではtanhを使う
* 判別機ではすべてLeakyReLUを使う

```
codings_size = 100

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
  tf.keras.layers.Reshape([7, 7, 128]),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="selu"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
])

discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, 
                         padding="same", activation=tf.keras.layers.LeakyReLU(0.2),
                         input_shape=[28, 28, 1]),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
                         activation=tf.keras.layers.LeakyReLU(0.2)),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid")
])
gan = tf.keras.models.Sequential([generator, discriminator])
discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.RMSprop())
discriminator.trainable = False
gan.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop())
```

訓練ループはGANと同じ

```
batch_size = 32

# チャンネル次元を追加し、tanhのために-1〜1の間の値となるよう変更する
scaled_X_train = X_train.reshape(-1, 28, 28, 1) * 2. - 1.

dataset = tf.data.Dataset.from_tensor_slices(scaled_X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, coding_size, n_epochs=50):
  generator, discriminator = gan.layers
  for epoch in range(n_epochs):
    for X_batch in dataset:
      X_batch = tf.cast(X_batch, dtype=tf.float32)

      # 判別器の訓練
      noise = tf.random.normal(shape=(batch_size, coding_size), dtype=tf.float32)
      generated_images = generator(noise)
      X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
      y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
      discriminator.trainable = True
      discriminator.train_on_batch(X_fake_and_real, y1)

      # 生成器の訓練
      noise = tf.random.normal(shape=(batch_size, coding_size))
      y2 = tf.constant([[1.]] * batch_size)
      discriminator.trainable = False
      l = gan.train_on_batch(noise, y2)
    print(epoch, l)
    
train_gan(gan, dataset, batch_size, codings_size)
```
