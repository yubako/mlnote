# YOLOX

物体検出のモデル

* Gitlab  
https://github.com/Megvii-BaseDetection/YOLOX

* Document  
https://yolox.readthedocs.io/en/latest/index.html

* 参考  
https://zenn.dev/opamp/articles/d3878b189ea256


## カスタムデータでファインチューニング

### dataset作成

[labelme](https://github.com/wkentaro/labelme)でannotationを作成し、train,valのディレクトリにそれぞれ格納する。  
※ COCO形式に変換するので、labelmeはpipでインストールするだけでなくcloneしておく必要がある

以下でCOCO形式に変換する

```
$ python3 ./labelme/examples/instance_segmentation/labelme2coco.py --labels ./labels.txt train/ train_dataset

$ python3 ./labelme/examples/instance_segmentation/labelme2coco.py --labels ./labels.txt val val_dataset
```

変換したものを以下のディレクトリ構成で配置する

```
./dataset/
|-- annotations
|   |-- instances_train.json
|   `-- instances_val.json
|-- train2017
|   `-- JPEGImages
|       `-- 01.jpg
`-- val2017
    `-- JPEGImages
        `-- 02.jpg
```

作成したデータセットをYOLOX/datasets/配下にコピーする

```
cp dataset/* YOLOX/datasets/.
```

### Expファイル作成

基底クラスは[yolox_base.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/exp/yolox_base.py)のExpで、このクラスを継承して必要な設定だけを行う。  
デフォルトで以下が用意されているので、コピーして使用する。

https://github.com/Megvii-BaseDetection/YOLOX/tree/main/exps/default

```
cp YOLOX/exps/default/yolox_s.py yolox_s.py
```

設定例
```
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # datasetのディレクトリ
        self.data_dir = "./datasets"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

        ## クラス数の変更
        self.num_classes = 1
	
        ## 評価間隔を変更（初期では10epochごとにしか評価が回らない）
        self.eval_interval = 1
```

### 学習

学習済みのモデルをダウンロードする

```
$ wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

学習実行
```
$ python tools/train.py \
    -f /path/to/your/Exp/file \
    -d 1 -b 64 --fp16 -o \
    -c /path/to/the/pretrained/weights [--cache]
```


