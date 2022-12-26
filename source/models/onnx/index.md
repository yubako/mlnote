# ONNX

pytorch, tensorflowなどで作ったmodelを、それらライブラリ外でも使えるようにしたライブラリ

参考  
https://qiita.com/studio_haneya/items/be9bc7c56af44b7c1e0a


pythonで使うためにインストールするには以下のようにする

```
pip install onnxruntime
```
or
```
pip install onnxruntime-gpu
```

GPU版はCUDAのバージョンで入れるものを変える必要がある  
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements


**公式サンプルソース**

https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb


## 使い方

モデルのload

```python
session = onnxruntime.InferenceSession("yolox_s.onnx")
```

デフォルトで使えるものを使用する場合

```python
from onnxruntime.datasets import get_example
exmodel = get_example("sigmoid.onnx")
sess = onnxruntime.InferenceSession(exmodel)
```

入力と出力層を確認

```python
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_type = sess.get_inputs()[0].type

print(input_name, input_shape, input_type)
---
x [3, 4, 5] tensor(float)
```

```
output_name = sess.get_outputs()[0].name
output_shape = sess.get_outputs()[0].shape
output_type = sess.get_outputs()[0].type

print(output_name, output_shape, output_type)
---
y [3, 4, 5] tensor(float)

```

推定

```
x = np.random.random(input_shape) * 6 - 3
x = x.astype(np.float32)

result = sess.run([output_name], {input_name: x})
```


