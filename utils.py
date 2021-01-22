import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def angular_error(gt, est):
    if len(gt) != len(est):
        print('The dimensions of estimation should be the same as GT.')
        return
    plt.plot(gt.ravel())
    plt.show()

    plt.plot(est.ravel())
    plt.show()
    dot = np.sum(gt*est, axis=1)

    dot[np.where(dot > 1)] = 1.0
    dot[np.where(dot < 0)] = 0.0

    plt.plot(dot)
    plt.title('dot')
    plt.show()

    ang_error_map = np.arccos(dot)*180.0 / np.pi

    plt.plot(ang_error_map)
    plt.show()

    print('Mean Angular Error : ', np.mean(ang_error_map))

    return normalize(ang_error_map)


def normalize(target):
    return (target - target.min()) / (target.max() - target.min())


def plot_normal(surface_normal, method) -> None:
    cv2.imshow(method, surface_normal)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return
