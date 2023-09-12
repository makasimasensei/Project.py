# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from graduation_project.Loss.det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,):
        super(DBLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, local_data, predicts):
        # num = len(local_data['img_path'])
        local_loss_all, local_loss_shrink_maps, local_loss_threshold_maps, local_loss_binary_maps = 0.0, 0.0, 0.0, 0.0
        for i in range(predicts.size(0)):
            img = local_data['image'][i]
            label_threshold_map = local_data['threshold_map'][i].to(self.device)
            label_threshold_mask = local_data['threshold_mask'][i].to(self.device)
            label_shrink_map = local_data['shrink_map'][i].to(self.device)
            label_shrink_mask = local_data['shrink_mask'][i].to(self.device)
            height, width, _ = img.shape

            predict_maps = F.interpolate(predicts, size=(height, width), mode='bilinear', align_corners=True)

            shrink_maps = predict_maps[i, 0, :, :]
            threshold_maps = predict_maps[i, 1, :, :]
            binary_maps = predict_maps[i, 2, :, :]

            loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,
                                             label_shrink_mask)
            loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,
                                               label_threshold_mask)
            loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map,
                                              label_shrink_mask)
            loss_shrink_maps = self.alpha * loss_shrink_maps
            loss_threshold_maps = self.beta * loss_threshold_maps

            loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
            local_loss_all += loss_all
            local_loss_shrink_maps += loss_shrink_maps
            local_loss_threshold_maps += loss_threshold_maps
            local_loss_binary_maps += loss_binary_maps
        # losses = {'loss': local_loss_all / num,
        #           "loss_shrink_maps": local_loss_shrink_maps / num,
        #           "loss_threshold_maps": local_loss_threshold_maps / num,
        #           "loss_binary_maps": local_loss_binary_maps / num}

        return local_loss_all
