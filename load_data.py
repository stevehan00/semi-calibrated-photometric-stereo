from scipy import io
import numpy as np
import cv2
import utils


def load_observations(obj_name, maximum):
    print('loading images ... ')

    base_dir = 'dataset/' + obj_name
    load_f_names = open(base_dir+'filenames.txt', 'r')

    observations = None
    cnt = 0

    for line in load_f_names:

        if cnt == maximum:
            break

        img = cv2.imread(base_dir + line[:-1], cv2.IMREAD_GRAYSCALE)

        if observations is None:
            observations = img.reshape((1, -1))
        else:
            observations = np.vstack((observations, img.reshape((1, -1))))

        cnt += 1

    print('completion!')
    observations = (observations - observations.min()) / (observations.max() - observations.min())
    return observations


def load_lights(obj_name, maximum):
    base_dir = 'dataset/' + obj_name
    load_lights = open(base_dir+'light_directions.txt', 'r')
    lights = []

    cnt = 0

    for line in load_lights:

        if cnt == maximum:
            break

        line = list(map(float, line[:-1].split(' ')))
        lights.append(line)
        cnt += 1

    lights = np.array(lights)
    return (lights-lights.min())/(lights.max()-lights.min())

import matplotlib.pyplot as plt

def load_normal_gt(obj_name):
    base_dir = 'dataset/' + obj_name
    load_normals = io.loadmat(base_dir + 'Normal_gt.mat')
    gt = np.array(load_normals['Normal_gt']).reshape((-1, 3))

    plt.plot(gt.ravel())
    plt.show()

    return gt

def load_intensities(obj_name):
    base_dir = 'dataset/' + obj_name
    load_intensities = open(base_dir + 'Normal_gt.mat', 'r')
    intensities = []

    for line in load_intensities:
        line = list(map(float, line[:-1].split(' ')))
        intensities.append(line)

    print('completion!')
    return np.array(intensities)
