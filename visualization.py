import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def save_image(mat, file_name):
    save_path = 'images/'
    plt.imshow(mat, vmin=0, vmax=1, cmap=plt.cm.Blues)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, format(mat[i][j], '.3f'), horizontalalignment="center", color="white")

    plt.colorbar()
    save_path += file_name
    plt.savefig(save_path)
    plt.close()