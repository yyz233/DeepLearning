import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import os


def draw_scatter(data, color):
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Scatter Plot')
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=10)


def draw_background(discriminator):
    x_min = y_min = -1.5
    x_max = y_max = 1.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i = x_min
    bg = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            bg.append([i, j])
            j += 0.01
        bg.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        bg.append([i, j])
        j += 0.01
        bg.append([i, y_max])
    bg.append([x_max, y_max])
    color = discriminator(torch.Tensor(bg).to(device))
    bg = np.array(bg)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0], bg[:, 1], c=np.squeeze(color.cpu().data), cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb


def gif_generate(path):
    imgs = []
    image_names = next(os.walk(path))[2]
    for image_name in image_names:
        imgs.append(imageio.imread(path + '/' + image_name))
    imageio.mimsave(path + "/final.gif", imgs, fps=5)