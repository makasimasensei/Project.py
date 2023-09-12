import json
import os
import random

import torch.functional
import torchvision

from graduation_project.Preprocessing.DecodeImage import *
from graduation_project.Preprocessing.MakeBorderMap import *
from graduation_project.Preprocessing.MakeShrinkMap import *
from torch.utils.data import Dataset

from graduation_project.Preprocessing.image_preprocessing import Normalize_image


class SimpleDataSet(Dataset):
    def __init__(self, mode, label_file, data_dir, seed=None):
        super(SimpleDataSet, self).__init__()
        # 标注文件中，使用'\t'作为分隔符区分图片名称与标签
        self.delimiter = '\t'
        # 数据集路径
        self.data_dir = data_dir
        # 随机数种子
        self.seed = seed
        # 获取所有数据，以列表形式返回
        self.data_lines = self.get_image_info_list(label_file)
        # 新建列表存放数据索引
        self.data_idx_order_list = list(range(len(self.data_lines)))
        self.normalize_image_fun = Normalize_image()
        self.mode = mode
        # 如果是训练过程，将数据集进行随机打乱
        if self.mode.lower() == "train":
            self.shuffle_data_random()

    @staticmethod
    def get_image_info_list(label_file):
        # 获取标签文件中的所有数据
        with open(label_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
        return lines

    def shuffle_data_random(self):
        # 随机打乱数据
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        # 1. 获取索引为idx的数据
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        # 2. 获取图片名称以及标签
        substr = data_line.strip("\n").split(self.delimiter)
        file_name = substr[0]
        label = substr[1]
        # 3. 获取图片路径
        img_path = os.path.join(self.data_dir, file_name)

        with open(img_path, 'rb') as f:
            img = f.read()
            decode_image = DecodeImage(img_mode='RGB', channel_first=False)
            img_decode = decode_image(img)

        label = json.loads(label)
        nbox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nbox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)
        data = {'img_path': img_path, 'image': img_decode}

        generate_text_border = MakeBorderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)
        data.update(generate_text_border(data, boxes, txt_tags))

        generate_shrink_map = MakeShrinkMap()
        data.update(generate_shrink_map(data, boxes, txt_tags))

        if self.mode.lower() == "train":
            if np.random.rand() < 0.5:
                random_num = random.randrange(3)
                if random_num == 0:
                    data = self.apply_flip(data)
                elif random_num == 1:
                    data = self.apply_rotation(data)
                elif random_num == 2:
                    data = self.apply_crop_zoom(data)

        data.update(self.normalize_image_fun(data))

        outs = data
        return outs

    def __len__(self):
        # 返回数据集的大小
        return len(self.data_idx_order_list)

    @staticmethod
    def expand_points_num(boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    @staticmethod
    def apply_flip(local_data):
        if random.randrange(2):
            for key, value in local_data.items():
                if key == 'img_path':
                    continue
                else:
                    local_data[key] = torch.flip(value, [1])
        else:
            for key, value in local_data.items():
                if key == 'img_path':
                    continue
                else:
                    local_data[key] = torch.flip(value, [0])
        return local_data

    @staticmethod
    def apply_rotation(local_data):
        fixed_angle = random.randint(-15, 15)
        for key, value in local_data.items():
            if key == 'img_path':
                continue
            elif key == 'image':
                transposed_tensor = value.permute(2, 0, 1)
                rotated_tensor = torchvision.transforms.functional.rotate(transposed_tensor, angle=fixed_angle)
                transposed_tensor = rotated_tensor.permute(1, 2, 0)
                local_data[key] = transposed_tensor
            else:
                new_tensor = value.unsqueeze(0)
                rotated_tensor = torchvision.transforms.functional.rotate(new_tensor, angle=fixed_angle)
                new_tensor = rotated_tensor.squeeze(0)
                local_data[key] = new_tensor

        return local_data

    @staticmethod
    def apply_crop_zoom(local_data):
        h, w, _ = local_data['image'].shape
        if h > 640 and w > 640:
            num = np.random.uniform(0.8, 0.9)
            center_crop = torchvision.transforms.CenterCrop((int(h * num), int(w * num)))
            for key, value in local_data.items():
                if key == 'img_path':
                    continue
                elif key == 'image':
                    transposed_tensor = value.permute(2, 0, 1)
                    crop_tensor = center_crop(transposed_tensor)
                    transposed_tensor = crop_tensor.permute(1, 2, 0)
                    local_data[key] = transposed_tensor
                else:
                    crop_tensor = center_crop(value)
                    local_data[key] = crop_tensor
        return local_data
