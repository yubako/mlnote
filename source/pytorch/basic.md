# pytorch

## Tensor基礎

* Tensor作成

```
a = torch.Tensor([1, 2, 3])
a = torch.arange(0, 10)
```

* 指定範囲を指定個数で作成

```
torch.linspace(0, 10, 5)
---
tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
---
```

* Numpyとの変換

```
a = torch.Tensor([1, 2, 3])
a = a.numpy()
a = torch.from_numpy(a)
```

* 形状変更

```
a = torch.tensor([[1,2,3], [4, 5, 6]])
print(a)
print(a.view(3, -1))
print(a.view(-1))
---
tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([[1, 2],
        [3, 4],
        [5, 6]])
tensor([1, 2, 3, 4, 5, 6])
```

* 次元の削減

```
a = torch.tensor([[1, 2, 3]])
print(a)
print(a.squeeze(dim=0))
---
tensor([[1, 2, 3]])
tensor([1, 2, 3])
```

* 次元の追加

```
a = torch.tensor([1, 2, 3])
print(a)
print(a.unsqueeze(dim=0))
print(a.unsqueeze(dim=1))
---
tensor([1, 2, 3])
tensor([[1, 2, 3]])
tensor([[1],
        [2],
        [3]])
```

## 活性化関数

* シグモイド関数

$$y = \frac{x}{1 + exp(-x)}$$

```
m = torch.nn.Sigmoid()
x = torch.linspace(-5, 5, 100)
y = m(x)
```

* ハイパボリックタンジェント関数

$$y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}$$

```
m = torch.nn.Tanh()
x = torch.linspace(-5, 5, 100)
y = m(x)
```

* ReLU

$$
y = \begin{cases}
0 ( x \leqq 0) \\
x ( x > 0)
\end{cases}
$$

```
m = torch.nn.ReLU()
x = torch.linspace(-5, 5, 100)
y = m(x)
```

* ソフトマックス関数

$$
y = \frac{exp(x)}{\sum_{k=1}^n exp(x_k)}
$$

```
m = torch.nn.Softmax(dim=1)
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])
y = m(x)
```

## 損失関数

* 平均二乗誤差

主に回帰問題に使用する

$$
E = \frac{1}{n}\sum_{k=1}^{n}(y_k - t_k)^2
$$

```
y = torch.tensor([3.0, 3.0, 3.0]) # 出力
t = torch.tensor([2.0, 1.0, 1.5]) # 正解ラベル

loss_func = torch.nn.MSELoss()
loss = loss_func(y, t)
loss.item()
```

* 交差エントロピー誤差

主に分類問題に使用する

$$
E = -\sum_k^nt_klog(y_k)
$$

```
# 出力
y = torch.tensor(
    [[3.0, 0.01, 0.1],
     [0.1, 3.0, 0.01]]
)

# 正解ラベル(One Hot Encodingにおける1の位置)
t = torch.tensor(
    [0, 1]
)

loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(y, t)
loss.item()
---
0.10012645274400711
```

## 最適化関数

* 確率的勾配降下法(SGD)

$$
w \larr w - \eta\frac{\partial E}{\partial w}
$$

$$
w = 重み \\
\eta = 学習係数
$$

```
optimizer = torch.optim.SGD(...)
```

* Momentum

SGD + 慣性

$$
w \larr w - \eta\frac{\partial E}{\partial w} + \alpha \Delta w
$$

$$
\alpha = 慣性の強さを決める定数 \\
\Delta w = 前回の更新量
$$

```
optimizer = torch.optim.SGD(..., momentum=0.9)
```

* AdaGrad

自動調整

$$
h \larr h + (\frac{\partial E}{\partial w})^2 $$
$$
w \larr w - \eta\frac{1}{\sqrt{h}}\frac{\partial E}{\partial w}
$$

更新のたびにhが更新されるため、更新するにつれてwの更新量は小さくなる。学習係数($\eta$)を持たないが、更新量が常に減少するので、途中でほぼ0になって更新されない(学習が進まない)ということが起こり得る。

```
optimizer = torch.optim.Adagrad(...)
```

* RMProp

AdaGradの更新量低下による停滞を克服

$$
h \larr \rho h + (1 - \rho)(\frac{\partial E}{\partial w})^2
$$
$$
w \larr w - \eta\frac{1}{\sqrt{h}}\frac{\partial E}{\partial w} 
$$

$\rho$により過去のhをある割合で忘却することで、更新量が低下しても再び学習が進むようになる

```
optimizer = torch.optim.RMSprop(...)
```

* Adam

いい感じ。まずはこれつかっとく

$$
m_0 = v_0 = 0
$$
$$
m_t = \beta_1m_{t-1} + (1 - \beta_1)\frac{\partial E}{\partial w}
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\frac{\partial E}{\partial w}
$$
$$
m'_t  = \frac{m_t}{1 - \beta_1^t}
$$
$$
v'_t = \frac{v_t}{1 - \beta_2^t}
$$
$$
w \larr w - \eta \frac{m'_t}{\sqrt{v'_t} + \epsilon}
$$

$\beta_1$, $\beta_2$, $\eta$, $\epsilon$が定数。tはパラメータ更新回数

```
optimizer = torch.optim.Adam(...)
```



$\mathbf{R_x} = \begin{bmatrix} 1 & 0 & 0 \ 0 & \cos{\alpha} & -\sin{\alpha} \ 0 & \sin{\alpha} & \cos{\alpha} \end{bmatrix}$


$\mathbf{R_y} = \begin{bmatrix} \cos{\beta} & 0 & \sin{\beta} \ 0 & 1 & 0 \ -\sin{\beta} & 0 & \cos{\beta} \end{bmatrix}$

$\mathbf{R_z} = \begin{bmatrix} \cos{\gamma} & -\sin{\gamma} & 0 \ \sin{\gamma} & \cos{\gamma} & 0 \ 0 & 0 & 1 \end{bmatrix}$

$\mathbf{R} = \mathbf{R_z} \mathbf{R_y} \mathbf{R_x}$

