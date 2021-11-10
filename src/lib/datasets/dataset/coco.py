from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import pandas as pd

import torch.utils.data as data

class COCO(data.Dataset):
  num_classes = 1
  default_resolution = [736, 1280]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(COCO, self).__init__()
    self.img_dir = '../../train_images'
    #import pdb;pdb.set_trace()
    if split == 'test':
        if opt.split_test != 'default':
            self.annot_path = opt.split_test
        else:
            self.annot_path = '../../work/coco_valid_fold0.json'
    elif split == 'val':
        if opt.split_val != 'default':
            self.annot_path = opt.split_val
        else:
            self.annot_path = '../../work/coco_valid_fold0.json'
    else:
        if opt.split_train != 'default':
            self.annot_path = opt.split_train
        else:
            self.annot_path = '../../work/coco_train_fold0.json'
    self.max_objs = 128
    self.class_name = ['__background__', 'helmet']
    self._valid_ids = [ 1, 2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    #import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          alpha = bbox[5]
          xdiff = bbox[6]
          ydiff = bbox[7]
          s = bbox[8]
          a = bbox[9]
          dis = bbox[10]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          file_name = self.coco.loadImgs(int(image_id))[0]['file_name']
          #57584_000336_Endzone_frame0010.jpg
          file_name = file_name.replace('.jpg','')
          video, frame = file_name.split('_frame')
          video_frame = f'{video}_{int(frame)}'

          detection = {
              "video_frame": video_frame,
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "left":   float("{:.2f}".format(bbox[0])),
              "top":    float("{:.2f}".format(bbox[1])),
              "width":  float("{:.2f}".format(bbox[2])),
              "height": float("{:.2f}".format(bbox[3])),
              "score":  float("{:.2f}".format(score)),
              "conf":   float("{:.2f}".format(score)),
              "alpha":  float("{:.2f}".format(alpha)),
              "xdiff":  float("{:.2f}".format(xdiff)),
              "ydiff":  float("{:.2f}".format(ydiff)),
              "s":      float("{:.2f}".format(s)),
              "a":      float("{:.2f}".format(a)),
              "dis":    float("{:.2f}".format(dis)),
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    #import pdb;pdb.set_trace()
    if self.opt.out_file_sufix == 'default':
      file_name = self.annot_path.split('/')[-1]
    else:
      file_name = self.annot_path.split('/')[-1].replace('.json','') + '-' + self.opt.out_file_sufix
    json_result = self.convert_eval_format(results)
    json.dump(json_result, open(f'{save_dir}/{file_name}.json', 'w'))
    pd.DataFrame(json_result).to_csv(f'{save_dir}/{file_name}.csv', index=False)
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    if self.opt.out_file_sufix == 'default':
      file_name = self.annot_path.split('/')[-1]
    else:
      file_name = self.annot_path.split('/')[-1].replace('.json','') + '-' + self.opt.out_file_sufix
    print(f'Saving results to {save_dir}/{file_name}.csv')
    coco_dets = self.coco.loadRes(f'{save_dir}/{file_name}.json')
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.params.iouThrs = np.array([0.35])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
