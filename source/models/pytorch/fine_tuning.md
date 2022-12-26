# ファインチューニング

出力層などを変更したモデルに対して、すべての層を学習する。

* 入力層に近い層のパラメータは学習率を小さくする
* 出力層に近い層のパラメータは学習率を大きくする

のが一般的

例：モデルの出力層を任意の出力に変更

```python
net = torchvision.models.vgg16(pretrained=True)
net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
net.train()
---
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=2, bias=True)
  )
)
```

学習するパラメータと学習率を設定する

```python
params_to_update_1 = []   # features層のパラメータ
params_to_update_2 = []   # classifiers層の付け替えていない層のパラメータ
params_to_update_3 = []   # 付け替えた層のパラメータ

update_param_names_1 = ["features"]
update_param_names_2 = [
    "classifier.0.weight", 
    "classifier.0.bias", 
    "classifier.3.weight", 
    "classifier.3.bias"
]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# パラメータごとに各リストに登録
for name, param in net.named_parameters():

  # `features`モデルのパラメータを格納
  if update_param_names_1[0] in name:
    param.requires_grad = True
    params_to_update_1.append(param)
  
  # `classifier`の最終層以外のパラメータを格納
  elif name in update_param_names_2:
    param.requires_grad = True
    params_to_update_2.append(param)
  
  # `classifier`の最終層のパラメータを格納
  elif name in update_param_names_3:
    param.requires_grad = True
    params_to_update_3.append(param)    

  else:
    # 学習しない
    param.requires_grad = False

# Optimizerを作成する際、それぞれのパラメータを指定する
optimizer = torch.optim.SGD([
    {"params": params_to_update_1, "lr": 1e-4},
    {"params": params_to_update_2, "lr": 5e-4},
    {"params": params_to_update_3, "lr": 1e-3},
], momentum=0.9)


```
