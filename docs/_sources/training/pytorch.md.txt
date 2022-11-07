# Pytorch

## 学習ループ

```python
import torch.nn as nn
import torch.optim as optim

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

record_loss_train = []
record_loss_test = []

for i in range(10):   # Epoch

  net.train()  # 学習モード
  loss_train = 0

  # Train
  for j, (data, label) in enumerate(ds_train):

    # GPU利用
    data.cuda()
    label.cuda()

    # Predict
    y = net(data)

    # 損失計算
    loss = loss_fn(y, label)

    # 勾配計算
    optimizer.zero_grad()
    loss.backward()

    # パラメータ更新
    optimizer.step()

    loss_train = loss.item()
    print("loss = ", loss_train / j)


  net.eval()  # validateモード
  loss_test = 0

  # Validate
  for j, (data, label) in enumerate(ds_test):

    # Predict
    y = net(data)

    # 損失計算
    loss = loss_fn(y, label)

    loss_test = loss.item()
    print("loss = ", loss_test / j)
```

