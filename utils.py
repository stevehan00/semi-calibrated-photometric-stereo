import numpy as np
import cv2
import matplotlib.pyplot as plt


def angular_error(gt, est, mask):
    """
    :param gt: ground truth of surface normal
    :param est: our estimations of surface normal
    :param mask: to mark object area in image
    :return: angular error map
    """

    if len(gt) != len(est):
        print('The dimensions of estimation should be the same as GT.')
        return

    angular_err = np.sum(gt*est, axis=1)
    plt.plot(angular_err)
    plt.show()
    angular_err[np.where(angular_err > 1)] = 1.0
    angular_err[np.where(angular_err < -1)] = -1.0

    ang_err_map = (np.arccos(angular_err)*180.0 / np.pi)
    mean_angular_err = np.mean(ang_err_map[np.where(mask == 1)])

    print('Mean Angular Error : ', mean_angular_err)

    return normalize(ang_err_map*mask)


def normalize(target):
    return (target - target.min()) / (target.max() - target.min())


def plot_normal(surface_normal, error_map, method):

    plt.subplot(1, 2, 1)
    plt.title('surface normal')
    plt.imshow(surface_normal[:, :, ::-1])
    plt.xticks([]); plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.title('angular error map')
    plt.imshow(error_map, cmap='gray')
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.show()
    return
