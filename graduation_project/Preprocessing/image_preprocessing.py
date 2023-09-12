import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


class Normalize_image:
    def __init__(self, scale=None, mean=None, std=None, order=None):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        # 1. 获得归一化的均值和方差
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        if isinstance(data, dict):
            img = data['image']
            img = img.numpy()
            if isinstance(img, Image.Image):
                img = np.array(img)
            assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        if isinstance(data, np.ndarray):
            img = data

        img_cv2 = cv2.resize(img, (640, 640))
        local_img = (img_cv2.astype('float32') * self.scale - self.mean) / self.std
        transposed_image = np.transpose(local_img, (2, 0, 1))
        four_dim_img = np.expand_dims(transposed_image, axis=0)
        tensor = torch.from_numpy(four_dim_img)
        dic = {'img_norm': tensor}
        return dic
