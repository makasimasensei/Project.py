import cv2
import numpy as np
import six
import torch


class DecodeImage(object):
    """ decode image """

    def __init__(self, img_mode='RGB', channel_first=False):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, img):
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        # 1. 图像解码
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)

        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % img.shape
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))
        data = torch.from_numpy(img.copy())
        return data
