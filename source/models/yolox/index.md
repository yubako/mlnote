# YOLOX

物体検出のモデル

* Gitlab  
https://github.com/Megvii-BaseDetection/YOLOX

* Document  
https://yolox.readthedocs.io/en/latest/index.html

* 参考  
https://zenn.dev/opamp/articles/d3878b189ea256



## pytorchモデルでの推論

以下を参考にする
* https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/demo.py
* https://www.kaggle.com/code/max237/getting-started-with-yolox-inference-only/notebook


```python
import torch
import os
import matplotlib.pyplot as plt
import time

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

class Predictor():

  def __init__(self, exp: Exp, path_to_checkpoint: str, class_names: list=COCO_CLASSES, device: str="gpu"):
    """
    推論クラス

    Arguments:
    ----------
      exp (Exp):
        yolox用Config(Experiment)のインスタンス
      
      path_to_checkpoint (str):
        学習済みモデルのチェックポイントファイルへのパス
      
      class_names (list):
        モデルに含まれる分類クラス名 (default: COCO_CLASSES)
      
      device (str):
        gpuを使用する場合に"gpu"を指定する
    """

    self.device = device
    self.model = exp.get_model()
    self.class_names = class_names
    self.num_classes = exp.num_classes
    self.confthre = exp.test_conf
    self.nmsthre = exp.nmsthre
    self.test_size = exp.test_size
    self.preproc = ValTransform()

    if self.device == "gpu":
      print(device)
      self.model.cuda()

    # This class is for prediction only, not for training
    self.model.eval()
    self.model.head.training=False
    self.model.training=False

    # Load in the weights
    print("loading checkpoint")
    if self.device == "gpu":
      chk = torch.load(path_to_checkpoint)
    else:
      chk = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
    self.model.load_state_dict(chk["model"])
    print("loaded checkpoint done.")

  def inference(self, img):
    """
    推論する

    Arguments:
    ----------
      img (str or np.ndarray):
        opencvで読み込んだimage、または画像ファイルへのパス

    Returns:
    --------
      outputs : モデルのpredict結果。以下のkeyを含むdict
        bboxes : bounding box
        classes : class names
        scores : scores

      img_info : 推論した画像情報。以下のkeyを含むdict
        id        : 0固定
        file_name : imgがファイルパスの場合にパスが格納される
        height    : 画像の高さ
        width     : 画像の幅
        raw_img   : 元の画像情報(np.ndarray)
        ratio     : モデルの入力サイズに合わせるために縮小した際のサイズ比
    """

    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
    img_info["ratio"] = ratio

    img, _ = self.preproc(img, None, self.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    if self.device == "gpu":
        img = img.cuda()

    with torch.no_grad():
        t0 = time.time()
        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
        print("Infer time: {:.4f}s".format(time.time() - t0))
    
    # convert output to bboxes
    output = outputs[0].cpu()
    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    return {"bboxes": bboxes, "classes": cls, "scores": scores}, img_info

  @classmethod
  def visualize(cls, img, bboxes, scores, classes, score_thr, class_names):
      return vis(img, bboxes, scores, classes, score_thr, class_names)
```

使い方

```python
predictor = Predictor(Exp(), "/content/yolox_x.pth", device="cpu")
out, info = predictor.inference("/content/beagle-hound-dog.jpg")
out
---
{'bboxes': tensor([[ 301.5980,  136.0976, 1327.3593,  962.3975],
         [ 337.3596,  328.7059,  447.2696,  376.1675],
         [ 369.8380,  332.1180,  447.2120,  370.2416]]),
 'classes': tensor([16., 32., 32.]),
 'scores': tensor([0.9727, 0.1161, 0.0252])}
```

visualize

```python
img_org = cv2.imread("/content/beagle-hound-dog.jpg")
img = predictor.visualize(img_org, out["bboxes"], out["scores"], out["classes"], 0.7, COCO_CLASSES)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```


## ONNXでの使用

### ONNX形式に変換

yolox_x.onnxに変換したモデルを保存する

```
$ python3 YOLOX/tools/export_onnx.py --output-name yolox_x.onnx -n yolox-x -c yolox_x.pth 
```

予め用意されているtoolsで推論してみる

```
$ wget https://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg
$ python3 YOLOX/demo/ONNXRuntime/onnx_inference.py -m yolox_x.onnx -i beagle-hound-dog.jpg -o output
```

### 推論について

デフォルトでは入力層の形は640,640なのでそのサイズにresizeが必要。幅高さの比率を変えてしまうとまずいのでpaddingを埋め込んでリサイズする。その比率と変換したイメージを作成するのが`yolox.data.data_augment.preproc`。戻り値は変換後のイメージと比率が返される。  
これにはバッチ要素が含まれていないので、次元をひとつ追加してINPUTとする

```
from yolox.data.data_augment import preproc as preprocess

origin_img = cv2.imread(args.image_path)
img, ratio = preprocess(origin_img, input_shape)
img = img[np.newaxis, :, :, :]  # (3, 640, 640) -> (1,3 , 640, 640)
```

また推論結果は(1, 8400, 85)の形を取る

```
outputs = sess.run(None, {"images": img})
print(outputs[0].shape)
---
(1, 8400, 85)
```

これらをbounding boxやscore値に変換する必要があるため、`demo_postprocess`を使用する  
参考(https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py)

```python
import torch
import os
import matplotlib.pyplot as plt
import time
import numpy as np

# Load yolox dependencies for inference later
COCO_CLASSES=["class1", "class2"]
import onnx, onnxruntime

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs



def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
    ]
)


class ONNXPredictor():

  def __init__(self, path_to_checkpoint: str, class_names: list=COCO_CLASSES):
    """
    推論クラス

    Arguments:
    ----------
      path_to_checkpoint (str):
        学習済みONNXモデルへのファイルパス
      
      class_names (list):
        モデルに含まれる分類クラス名 (default: COCO_CLASSES)
    """

    self.class_names = class_names
    self.num_classes = len(class_names)
    self.confthre = 0.25
    self.nmsthre = 0.45

    # Load in the weights
    print("loading checkpoint")
    self.session = onnxruntime.InferenceSession(path_to_checkpoint)
    self.input_name = self.session.get_inputs()[0].name
    self.input_shape = self.session.get_inputs()[0].shape[2:]
    print("loaded checkpoint done.")


  def inference(self, img):
    """
    推論する

    Arguments:
    ----------
      img (str or np.ndarray):
        opencvで読み込んだimage、または画像ファイルへのパス

    Returns:
    --------
      outputs : モデルのpredict結果。以下のkeyを含むdict
        bboxes : bounding box
        classes : class names
        scores : scores

      img_info : 推論した画像情報。以下のkeyを含むdict
        id        : 0固定
        file_name : imgがファイルパスの場合にパスが格納される
        height    : 画像の高さ
        width     : 画像の幅
        raw_img   : 元の画像情報(np.ndarray)
        ratio     : モデルの入力サイズに合わせるために縮小した際のサイズ比
    """

    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    #ratio = min(self.input_shape[0] / img.shape[0], self.input_shape[1] / img.shape[1])
    #img_info["ratio"] = ratio
    #img, _ = self.preproc(img, None, self.input_shape)
    #img = img[np.newaxis, :, :, :]
    #img = img.astype(np.float32)

    t0 = time.time()

    # Reference
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
    # Lisence :  Apache-2.0 license
    img, ratio = preproc(img, self.input_shape)
    img_info["ratio"] = ratio
    img = img[np.newaxis, :, :, :]

    outputs = self.session.run(None, {self.input_name: img})
    predictions = demo_postprocess(outputs[0], self.input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmsthre, score_thr=self.confthre)
    if dets is not None:
      out = dict(
          bboxes=dets[:, :4], scores=dets[:, 4], classes=dets[:, 5]
      )
    else:
      out = dict(
          bboxes=[], scores=[], classes=[]
      )

    print("Infer time: {:.4f}s".format(time.time() - t0))
    return out, img_info, dets

  @classmethod
  def visualize(cls, img, bboxes, scores, classes, score_thr, class_names):
      return vis(img, bboxes, scores, classes, score_thr, class_names) 
```

使い方


```python
predictor = ONNXPredictor("/content/yolox_x.onnx")
out, info = predictor.inference("/content/beagle-hound-dog.jpg")
out
---
loading checkpoint
loaded checkpoint done.
Infer time: 6.9102s
{'bboxes': array([[ 301.59790039,  136.09751892, 1327.35913086,  962.39746094]]),
 'scores': array([0.97268456]),
 'classes': array([16.])}
```

visualize

```python
img_org = cv2.imread("/content/beagle-hound-dog.jpg")
img = predictor.visualize(img_org, out["bboxes"], out["scores"], out["classes"], 0.5, COCO_CLASSES)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```


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

        ## エポック数は控えめ
        self.max_epoch = 30

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


