from torch.utils.data import DataLoader
from graduation_project.LoadData.SimpleDataSet import *


def collate_fn(batch):
    # 自定义转换操作
    # 这里以处理不定长序列为例
    img_path, image, img_norm, threshold_map, threshold_mask, shrink_map, shrink_mask = [], [], [], [], [], [], []
    for i in range(len(batch)):
        img_path.append(batch[i]['img_path'])
        image.append(batch[i]['image'])
        img_norm.append(batch[i]['img_norm'])
        threshold_map.append(batch[i]['threshold_map'])
        threshold_mask.append(batch[i]['threshold_mask'])
        shrink_map.append(batch[i]['shrink_map'])
        shrink_mask.append(batch[i]['shrink_mask'])
    img_norm_cat = torch.cat(img_norm, dim=0)

    data = {'image': image, 'img_path': img_path, 'img_norm': img_norm_cat,
            'threshold_map': threshold_map, 'threshold_mask': threshold_mask,
            'shrink_map': shrink_map, 'shrink_mask': shrink_mask}

    return data


def build_dataloader(mode, label_file, data_dir, batch_size, num_workers, seed=None):
    # 创建数据读取类
    dataset = SimpleDataSet(mode, label_file, data_dir, seed)
    # 使用DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             num_workers=num_workers, collate_fn=collate_fn, drop_last=True)

    return data_loader
