from train import *
from timefun import *


def trainer(architecture, local_input_path, local_epoch, local_checkpoints, local_pretrained):
    input_path = local_input_path
    epochs = local_epoch
    if local_checkpoints is not None:
        checkpoint = torch.load(local_checkpoints)
        current_epoch = checkpoint['epoch'] + 1
        current_loss = checkpoint['loss']
        current_eval = checkpoint['loss_eval']
        print('-----训练轮次是：-----{}'.format(current_epoch))
        print('-----训练集的损失函数是：-----{}'.format(current_loss))
        print('-----测试集的损失函数是：-----{}'.format(current_eval))
    else:
        current_epoch = 0
    runtime = 0
    start_train = train(architecture, input_path, local_checkpoints, local_pretrained)
    for epoch in range(0, epochs - current_epoch):
        start_time = time.time()
        print("-----第{}轮训练开始-----".format(current_epoch + epoch + 1))
        start_train(current_epoch + epoch)
        end_time = time.time()
        total_time = end_time - start_time
        runtime += total_time
        eta = (runtime / (epoch + 1)) * (epochs - epoch - current_epoch)
        print("第{}轮训练时间是：{}".format(epoch + current_epoch + 1, total_time))
        time_fun(int(eta))
