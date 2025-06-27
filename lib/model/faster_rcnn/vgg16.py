# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    # support for Caffe and PyTorch models
    self.model_path_caffe = 'data/pretrained_model/vgg16_caffe.pth'
    self.model_path_pytorch = 'data/pretrained_model/vgg16_pytorch.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    #vgg = models.vgg16()
    vgg = models.vgg16(pretrained=False)

    if self.pretrained:
      loaded = False

      # try loading Caffe
      if os.path.exists(self.model_path_caffe):
        try:
          print("Loading Caffe pretrained weights from %s" % self.model_path_caffe)
          state_dict = torch.load(self.model_path_caffe)
          vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
          loaded = True
        except Exception as e:
          print(f"Failed to load Caffe model: {e}")

      # Try to load PyTorch model if exists
      if not loaded and os.path.exists(self.model_path_pytorch):
        try:
          print("Loading PyTorch pretrained weights from %s" % self.model_path_pytorch)
          state_dict = torch.load(self.model_path_pytorch)
          vgg.load_state_dict(state_dict)
          loaded = True
        except Exception as e:
          print(f"Failed to load PyTorch model: {e}")

      if not loaded:
        print("No local pretrained model found. Loading PyTorch ImageNet pretrained weights...")
        pretrained_vgg = models.vgg16(pretrained=True)
        vgg.load_state_dict(pretrained_vgg.state_dict())
        loaded = True

        print(f"Saving PyTorch pretrained weights to {self.model_path_pytorch}")
        os.makedirs(os.path.dirname(self.model_path_pytorch), exist_ok=True)
        torch.save(pretrained_vgg.state_dict(), self.model_path_pytorch)

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

