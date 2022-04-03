# Tensorflow model

## 基本的なmodelの使い方

```
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])
model.build(input_shape=(None, 10))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)
model.fit(x, y)
```

## Datasetを使用したmodelの学習

datasetを使用する場合はデータ数を自動的に取得できないので、steps_per_epochの指定が必要
```
epochs = 100
batch_size = 5
steps_per_epoch = np.ceil(X_train.shape[0] / batch_size)

train = input_fn()
valid = input_fn_eval()
hist = model.fit(train, validation_data=valid,
                 batch_size=batch_size, epochs=epochs, 
                 verbose=0, steps_per_epoch=steps_per_epoch)
```

## Estimator

* modelを作成
```
model = tf.keras.models.Sequential([...])
model.compile(...)
```

* 入力関数を作成
```
from sklearn.impute import SimpleImputer
import tensorflow as tf

def input_fn():
    imr = SimpleImputer(missing_values=np.nan, strategy="mean")

    x_train = pd.get_dummies(train, columns=["Sex", "Embarked"])
    y_train = x_train["Survived"]
    x_train = x_train.drop(["PassengerId", "Name", "Cabin", "Ticket", "Survived"], axis=1)

    imr.fit(x_train)
    x_train_imr = imr.transform(x_train)
    x_train_imr.shape
    ds = tf.data.Dataset.from_tensor_slices((x_train_imr, y_train))
    ds = ds.shuffle(100).batch(5).repeat()
    return ds

data = next(iter(input_fn().take(1)))
print(data[0].shape)
print(data[1])
```

* keras modelからestimatorを作成し、学習する

```
import tempfile
model_dir = tempfile.mkdtemp()
estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)
estimator.train(input_fn=input_fn, steps=1000)
eval = estimator.evaluate(input_fn=input_fn, steps=10)
```


