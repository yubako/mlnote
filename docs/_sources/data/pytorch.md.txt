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
