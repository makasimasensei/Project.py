import os

import matplotlib.pyplot as plt
import torch
from graduation_project.Architecture.architecture import MyNet
from graduation_project.LoadData.load_data import MyDataloader
from graduation_project.Loss.loss import DBLoss


def eval_mode(local_input_path, local_checkpoints):
    loss = 0
    local_train_data_label_ = os.path.join(local_input_path, "train.txt")
    local_eval_data_label_ = os.path.join(local_input_path, "eval.txt")
    mynet = MyNet().to('cuda')
    db_loss = DBLoss().to('cuda')
    checkpoint = torch.load(local_checkpoints)
    mynet.load_state_dict(checkpoint['dic'])
    local_eval_dataloader = (MyDataloader(
        local_input_path, local_train_data_label_, local_eval_data_label_).
                             eval_dataloader_fun(batch_size=4, num_workers=4))
    with torch.no_grad():
        for data in local_eval_dataloader:
            input_x = data['img_norm'].to('cuda')
            output_mynet = mynet(input_x).to('cuda')
            # plt.subplot(2, 3, 1)
            # plt.imshow(data['image'][0])
            # plt.subplot(2, 3, 2)
            # img1 = output_mynet[0, 0, :, :]
            # plt.imshow(img1.cpu().detach().numpy())
            # plt.subplot(2, 3, 3)
            # img2 = output_mynet[0, 1, :, :]
            # plt.imshow(img2.cpu().detach().numpy())
            # plt.subplot(2, 3, 4)
            # img3 = output_mynet[0, 2, :, :]
            # plt.imshow(img3.cpu().detach().numpy())
            # plt.show()

            loss_all = db_loss(data, output_mynet)
            loss += loss_all

    return loss / len(local_eval_dataloader)


if __name__ == '__main__':
    input_path = "E:/anaconda/envs/pytorch/graduation_project/demo"
    check = "E:/anaconda/envs/pytorch/graduation_project/checkpoints/model_weights_1200.pth"
    print(eval_mode(input_path, check))
