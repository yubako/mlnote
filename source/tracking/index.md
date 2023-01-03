# Tracking

物体追跡メモ

物体追跡モデル

* ByteTrack https://github.com/ifzhang/ByteTrack

  誤検知を抑えることと検知率の低下を防ぐことをいい感じに行って、障害物等でdetectorの結果が信頼度の低いものであっても追跡できるようにしたもの。

* OC-SORT https://github.com/noahcao/OC_SORT

  物体の重なり(オクルージョン)や非線形な動きをする物体のトラッキングをよりロバストにしたもの。


## ByteTrack

以下を参考にして、YOLOXから得られたbbox, scoreをByteTrackerに与えることで追跡する
https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/evaluators/mot_evaluator.py

```python
from yolox.tracker.byte_tracker import BYTETracker
import cv2
from google.colab.patches import cv2_imshow

#    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
#    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
#    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

class Args():
  track_thresh = 0.3
  track_buffer = 30
  match_thresh = 0.8
  min_box_area = 50
  mot20 = False

args = Args()
tracker = BYTETracker(args)

cap = cv2.VideoCapture("myvideo.mp4")
results = []

def drawbox(img, id, x1, y1, x2, y2):
  img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 100), thickness=2, lineType=cv2.LINE_4)
  img = cv2.putText(img, f"{id}", (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4)
  return img


for i in range(2):

  r, img = cap.read()
  out, img_info, dets = predictor.inference(img)
  bboxes = out["bboxes"]
  scores = out["scores"]
  classes = out["classes"]
  dets = [[*box, score] for box, score in zip(bboxes, scores)]
  dets = np.array(dets)

  online_targets = tracker.update(dets, img.shape[:2], img.shape[:2])
  online_tlwhs = []
  online_ids = []
  online_scores = []
  im = img.copy()
  for t in online_targets:
      tlwh = t.tlwh
      tid = t.track_id
      vertical = tlwh[2] / tlwh[3] > 1.6
      if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
          online_tlwhs.append(tlwh)
          online_ids.append(tid)
          online_scores.append(t.score)
          im = drawbox(im, tid, int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
  # save results
  results.append((i, online_tlwhs, online_ids, online_scores))
  cv2_imshow(im)
```

## OC-SORT


```python
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer

    # tracking args
#    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#    parser.add_argument("--public", action="store_true", help="use public detection")
#    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
#    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.)
#
#    parser.add_argument(
#        "--aspect_ratio_thresh", type=float, default=1.6,
#        help="threshold for filtering out boxes of which aspect ratio are above the given value."
#    )
#    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return 

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


class Args():
  track_thresh = 0.6
  iou_thresh = 0.3
  min_hits = 3
  inertia = 0.2
  deltat = 3
  track_buffer = 30
  match_thresh = 0.9
  min_box_area = 100
  gt_type="_val_half"
  mot20 = False
  asso="iou"
  use_byte=False
  aspect_ratio_thresh = 1.6
  min_box_area=10

cap = cv2.VideoCapture("myvideo.mp4")

args = Args()
tracker = OCSort(det_thresh = args.track_thresh, iou_threshold=args.iou_thresh,
    asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia)

results = []
for frame_id in range(3):
  r, img = cap.read()
  if r == False:
    break
  output, img_info, dets = predictor.inference(img)

  #imh = predictor.visualize(img_info["raw_img"], info["bboxes"], info["scores"], info["classes"], 0.7, ["duck"])
  dets = [[*box, score] for box, score in zip(output["bboxes"], output["scores"])]
  dets = np.array(dets)

  online_targets = tracker.update(dets, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']],)
  online_tlwhs = []
  online_ids = []
  img = img_info["raw_img"].copy()
  for t in online_targets:
      tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
      tid = t[4]
      vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
      if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
        online_tlwhs.append(tlwh)
        online_ids.append(tid)
        print(
            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1"
        )
        #img = drawbox(img, tid, int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
  img = plot_tracking(
      img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1.
  )

  cv2_imshow(img)
  cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
```

