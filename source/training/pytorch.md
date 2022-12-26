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

関数化例

```python
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

  # epochのループ
  for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    print("------")

    # epochごとの学習と検証ループ
    for phase in ["train", "val"]:
      if phase == "train":
        net.train()
      else:
        net.eval()
      
      epoch_loss = 0.0    # epochごとの損失和
      epoch_corrects = 0  # epochごとの正解数

      # 未学習時の検証性能を確かめるため、epoch=0の学習は省略
      if (epoch == 0) and (phase == "train"):
        continue
   
      # データローダーからミニバッチを取得
      for inputs, labels in tqdm(dataloaders_dict[phase]):

        # 初期化
        optimizer.zero_grad()

        # 順伝搬の計算
        with torch.set_grad_enabled(phase == "train"):
          outputs = net(inputs)
          loss = criterion(outputs, labels)  # 損失計算
          _, preds = torch.max(outputs, 1)   # ラベルを予測

          # 訓練時のみバックプロパゲーション
          if phase == "train":
            loss.backward()
            optimizer.step()
        
          # イテレーションの計算
          # lossの合計を計算 (loss.item()はバッチの平均が得られるので、合計を求めるために要素数をかける)
          epoch_loss += loss.item() * inputs.size(0)

          # 正解数の合計を更新
          epoch_corrects += torch.sum(preds == labels.data)
      
    # epochごとのloss, 正解率を表示(epoch 平均)
    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
    epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

    print("{}: loss: {:.4f}, Acc: {:4f}".format(phase, epoch_loss, epoch_acc))

```


