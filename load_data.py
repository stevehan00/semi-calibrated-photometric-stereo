from scipy import io
import numpy as np
import cv2


def load_observations(obj_name):
    print('loading images ... ')

    base_dir = 'dataset/' + obj_name
    load_f_names = open(base_dir+'filenames.txt', 'r')

    observations = None

    for line in load_f_names:
        img = cv2.imread(base_dir + line[:-1], cv2.IMREAD_GRAYSCALE)

        if observations is None:
            observations = img.reshape((1, -1))
        else:
            observations = np.vstack((observations, img.reshape((1, -1))))
    print('completion!')
    observations = (observations - observations.min()) / (observations.max() - observations.min())
    return observations


def load_lights(obj_name):
    base_dir = 'dataset/' + obj_name
    load_lights = open(base_dir+'light_directions.txt', 'r')
    lights = []

    for line in load_lights:
        line = list(map(float, line[:-1].split(' ')))
        lights.append(line)

    lights = np.array(lights)
    return (lights-lights.min())/(lights.max()-lights.min())


def load_normal(obj_name):
    base_dir = 'dataset/' + obj_name
    load_normals = io.loadmat(base_dir + 'Normal_gt.png')

    return load_normals


def load_intensities(obj_name):
    base_dir = 'dataset/' + obj_name
    load_intensities = open(base_dir + 'Normal_gt.mat', 'r')
    intensities = []

    for line in load_intensities:
        line = list(map(float, line[:-1].split(' ')))
        intensities.append(line)

    print('completion!')
    return np.array(intensities)
