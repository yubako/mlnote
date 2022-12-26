# pytorch

## DataLoaderの使い方

```python
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

mnist_train = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(mnist_train)

'''
<class 'torchvision.datasets.mnist.MNIST'>
Dataset MNIST
    Number of datapoints: 60000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: ToTensor()
'''

# DataLoader作成
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

fig = plt.figure()
batch_no = 0
for i, data in enumerate(train_loader):
  if i >= 5: 
    break
  ax = fig.add_subplot(1, 5, i+1)
  ax.imshow(data[0][batch_no].view(28, 28))
  ax.set_title(data[1][batch_no])

plt.show()
```

## 画像処理

transformsを使って必要に応じた前処理を行う

pytorchで使う画像は(C, H, W)に対して、matplotlibなどは(H, W, C)な点に注意が必要

```python
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torchvision 
from torchvision import models, transforms

class BaseTransform():
  """
  画像サイズをリサイズし、色を標準化する

  Attributes
  ----------
  resize: int
    リサイズ先の画像の大きさ
  
  mean: (R, G, B)
    各色のチャンネル平均値
  
  std: (R, G, B)
    各色チャネルの標準偏差
  """
  def __init__(self, resize, mean, std):
    self.base_transform = transforms.Compose([
        transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
        transforms.CenterCrop(resize),  # 画像中央をresize x resizeで切り取る
        transforms.ToTensor(),  # Tensorに変換
        transforms.Normalize(mean, std) # 色情報の標準化
    ])

  def __call__(self, img):
    return self.base_transform(img)

# 動作確認
image_file_path = '/content/pytorch_advanced/1_image_classification/data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

# 元画像の表示
plt.imshow(img)
plt.show()

# 画像の前処理と処理済み画像の表示
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.223, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

# (C, H, W) -> (H, W, C)に変換
img_transformed = img_transformed.numpy().transpose((1, 2, 0))

# 色を0-1に制限
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()
```

## 独自のDataSetの作り方

データを１つ取り出す`__getitem__()`とデータの数を返す`__len__()`を実装する必要がある

```python
class HymenopteraDataset(data.Dataset):
  """
  ありとハチの画像のDatasetクラス。PyTorchのDataSetクラスを継承

  Attriutes
  ---------
  file_list : リスト
    画像のパスを格納したリスト
  
  transform: object
    前処理クラスのインスタンス
  
  phase : "train" or "val"
  """
  def __init__(self, file_list, transform=None, phase="train"):
    self.file_list = file_list
    self.transform = transform
    self.phase = phase
  
  def __len__(self):
    """ 画像の枚数を返す """
    return len(self.file_list)
  
  def __getitem__(self, index):
    """
    前処理をおこなった画像のTensor形式のデータとラベルを取得
    """
    img_path = self.file_list[index]
    img = Image.open(img_path)

    # 前処理
    img_transformed = self.transform(img, self.phase) # torchSize([3, 224, 224])

    if "ant" in img_path:
      label = 0
    elif "bees" in img_path:
      lael = 1
    
    return img_transformed, label
```




