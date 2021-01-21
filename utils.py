import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def angular_error(gt, est):
    """
    gt, est : p*3
    """

    if len(gt) != len(est):
        print('The dimensions of estimation should be the same as GT.')

    mean_ang_error = 0.0
    temp = []
    cnt = 0

    for i in range(len(gt)):
        numerator = sum(gt[i]*est[i])
        denominator = np.sqrt(gt[i][0]**2 + gt[i][1]**2 + gt[i][2]**2) * \
            np.sqrt(est[i][0]**2 + est[i][1]**2 + est[i][2]**2)
        # exceptions
        if denominator == 0:
            continue
        cnt += 1
        # angular between two vectors
        ang_error = (math.acos(numerator/denominator)*180) / math.pi
        mean_ang_error += ang_error
        temp.append(ang_error)

    mean_ang_error /= cnt

    plt.plot(temp)
    plt.show()

    print('mean angular error :', mean_ang_error)
    return


def normalize(target):
    return (target - target.min()) / (target.max() - target.min())


def plot_normal(surface_normal, method) -> None:
    cv2.imshow(method, surface_normal)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return
