from matplotlib import pyplot as plt
from graduation_project.LoadData.build_dataloader import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MyDataloader:
    def __init__(self, ic15_data_path_, train_data_label_, eval_data_label_):
        self.ic15_data_path = ic15_data_path_
        self.train_data_label = train_data_label_
        self.eval_data_label = eval_data_label_

    def train_dataloader_fun(self, mode='train', batch_size=4, num_workers=4):
        # 定义训练集数据读取器，进程数设置为8
        _train_dataloader = build_dataloader(mode, self.train_data_label, self.ic15_data_path, batch_size, num_workers)
        return _train_dataloader

    def eval_dataloader_fun(self, mode='Eval', batch_size=4, num_workers=4):
        # 定义验证集数据读取器
        _eval_dataloader = build_dataloader(mode, self.eval_data_label, self.ic15_data_path, batch_size, num_workers)
        return _eval_dataloader


if __name__ == "__main__":
    ic15_data_path = "E:/anaconda/envs/pytorch/graduation_project/demo"
    train_data_label = "E:/anaconda/envs/pytorch/graduation_project/demo/train.txt"
    eval_data_label = "E:/anaconda/envs/pytorch/graduation_project/demo/eval.txt"
    # train_dataloader = \
    #     (MyDataloader(ic15_data_path, train_data_label, eval_data_label).train_dataloader_fun(batch_size=4))
    # for i in train_dataloader:
    #     print(i[0])

    local_eval_dataloader = (MyDataloader(
        ic15_data_path, train_data_label, eval_data_label).eval_dataloader_fun(batch_size=4, num_workers=4))
    for i in local_eval_dataloader:
        print(i['img_path'][0])
