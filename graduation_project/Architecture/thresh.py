import matplotlib.pyplot as plt
import numpy as np


def show_binarize_map(x):
    numpy_array = x.detach().numpy()
    squeezed_arr = np.squeeze(numpy_array)
    transposed_arr = np.transpose(squeezed_arr, (1, 2, 0))
    plt.imshow(transposed_arr)
    plt.axis('off')  # 关闭坐标轴
    plt.show()