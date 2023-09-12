import json
from graduation_project.Preprocessing.DecodeImage import *


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


def read_data(local_data):
    boxes_list, txts_list, txt_tags_list = [], [], []
    img_path, label, image = [], [], []

    for i in range(len(local_data[0])):
        img_name = local_data[0][i]
        gt_label = local_data[1][i]
        img_path.append(img_name)
        label.append(gt_label)

        # 4. 声明DecodeImage类，解码图像
        decode_image = DecodeImage(img_mode='RGB', channel_first=False)
        data = decode_image(local_data[2][i])
        if data is None:
            return None
        else:
            image.append(data)

        label = json.loads(gt_label)
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
        boxes = expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)
        boxes_list.append(boxes)
        txts_list.append(txts)
        txt_tags_list.append(txt_tags)
    data = {'image': image, 'label': label, 'img_path': img_path,
            'polys': boxes_list, 'texts': txts_list, 'ignore_tags': txt_tags_list}
    return data
