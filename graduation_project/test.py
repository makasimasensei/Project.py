import os

import cv2
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg

from graduation_project.Architecture.architecture import MyNet
from graduation_project.PostProcess.postprocess import SegDetectorRepresenter
from graduation_project.Preprocessing.image_preprocessing import Normalize_image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def demo_mode(local_checkpoints, img_path):
    mynet = MyNet().to('cuda')
    checkpoint = torch.load(local_checkpoints)
    mynet.load_state_dict(checkpoint['dic'])
    img = cv2.imread(img_path)
    norm = Normalize_image()
    image = norm(img)
    input_x = image['img_norm'].to('cuda')
    output_mynet = mynet(input_x)

    db_pp = SegDetectorRepresenter()
    result = db_pp([1080, 760], output_mynet, is_output_polygon=False)
    return result


def show_img(image, dic):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(len(dic[0][0])):
        my_pic = fin[0][0][i]
        # 循环遍历每个线段并绘制
        x, y = [], []
        for segment in my_pic:
            x.append(segment[0])
            y.append(segment[1])
        x.append(my_pic[0][0])
        y.append(my_pic[0][1])
        ax.plot(x, y, c='r', linestyle='-', label='Connect')
    plt.show()


if __name__ == '__main__':
    check = "E:/anaconda/envs/pytorch/graduation_project/checkpoints/model_weights_1200.pth"
    pic = "C:/Users/14485/Pictures/Screenshots/gt_0.jpg"
    image_read = mpimg.imread(pic)
    fin = demo_mode(check, pic)
    show_img(image_read, fin)
