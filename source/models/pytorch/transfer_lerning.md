# 転移学習

出力層を付け替える

```python
from torchvision import models, transforms
net = models.vgg16(pretrained=True)
print(net.classifier)

---
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace=True)
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=4096, out_features=1000, bias=True) <-- デフォルト1000種
)
---
# 2種に差し替える
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
net.train()
---
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=2, bias=True)  <--
  )
---
```

学習中、出力層以外のパラメータは更新せず、変更したレイヤーのパラメータだけを更新する  
更新するかどうかは`requires_grad`で設定する

```
# 転移学習で学習させるパラメータを格納
params_to_update = []

# 学習させるパラメータ名
update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():   # パラメータを名前付きで全部取得
  if name in update_param_names:
    param.requires_grad = True
    params_to_update.append(param)
  else:
    param.requires_grad = False
```

得られたparams_to_updateをoptimizerに渡す

```
optimizer = torch.optim.SGD(params=params_to_update)
```


