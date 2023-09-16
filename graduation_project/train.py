import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append("/graduation_project/LoadData")
sys.path.append("/graduation_project/Preprocessing")
sys.path.append("/graduation_project/Loss")
sys.path.append("/graduation_project/Architecture")

import time
import os

import torch
from graduation_project.Architecture.architecture import MyNet
from graduation_project.LoadData.load_data import MyDataloader
from graduation_project.Loss.loss import DBLoss


class train:
    def __init__(self, architecture, path, local_checkpoints, local_pretrained):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_ic15_data_path = path
        self.local_train_data_label_ = os.path.join(self.local_ic15_data_path, "train.txt")
        self.local_eval_data_label_ = os.path.join(self.local_ic15_data_path, "eval.txt")
        self.local_eval_dataloader = (MyDataloader(
            self.local_ic15_data_path, self.local_train_data_label_, self.local_eval_data_label_).
                                 eval_dataloader_fun(batch_size=4, num_workers=4))
        self.mynet = MyNet(architecture).to(self.device)
        self.db_loss = DBLoss().to(self.device)
        self.loss = 0
        self.loss_eval = 0
        if local_pretrained is not None:
            checkpoint = torch.load(local_pretrained)
            self.mynet.load_state_dict(checkpoint['dic'])
        elif local_checkpoints is not None:
            checkpoint = torch.load(local_checkpoints)
            self.mynet.load_state_dict(checkpoint['dic'])
        self.optim = torch.optim.Adam(self.mynet.parameters(), lr=0.001, weight_decay=0.001)

    def __call__(self, local_epoch):
        bk = 0
        local_train_dataloader = (MyDataloader(
            self.local_ic15_data_path, self.local_train_data_label_, self.local_eval_data_label_).
                                  train_dataloader_fun(batch_size=4, num_workers=4))
        for data in local_train_dataloader:
            start_time = time.time()
            self.train_processing(data)
            end_time = time.time()
            bk += end_time - start_time
        if (local_epoch + 1) % 10 == 0:
            with torch.no_grad():
                for data in self.local_eval_dataloader:
                    self.loss_eval += self.eval_processing(data)
                print('-----测试集的损失函数是：-----{}'.format(self.loss_eval / len(self.local_eval_dataloader)))
            torch.save({'epoch': local_epoch, 'dic': self.mynet.state_dict(),
                        'loss': self.loss / len(local_train_dataloader),
                        'loss_eval': self.loss_eval / len(self.local_eval_dataloader)},
                       'E:/anaconda/envs/pytorch/graduation_project/checkpoints'
                       '/model_weights_{}.pth'.format(local_epoch + 1))
        self.loss_eval = 0
        print(self.loss / len(local_train_dataloader))
        print("第一段运行时间是：{}".format(bk))
        self.loss = 0

    def train_processing(self, data_list):
        input_x = data_list['img_norm'].to(self.device)
        output_mynet = self.mynet(input_x).to(self.device)
        # plt.figure(figsize=(8, 8))
        # plt.subplot(2, 2, 1)
        # image = data_list['image'][0]
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # img1 = output_mynet[0, 0, :, :]
        # plt.imshow(img1.cpu().detach().numpy())
        # plt.subplot(2, 2, 3)
        # img2 = output_mynet[0, 1, :, :]
        # plt.imshow(img2.cpu().detach().numpy())
        # plt.subplot(2, 2, 4)
        # img3 = output_mynet[0, 2, :, :]
        # plt.imshow(img3.cpu().detach().numpy())
        # plt.tight_layout()
        # plt.show()

        loss_all = self.db_loss(data_list, output_mynet)
        self.loss += loss_all
        self.optim.zero_grad()
        loss_all.backward()
        self.optim.step()

    def eval_processing(self, data_list):
        input_x = data_list['img_norm'].to(self.device)
        output_mynet = self.mynet(input_x).to(self.device)

        loss_all = self.db_loss(data_list, output_mynet)
        return loss_all
