import load_data, utils
import cv2
import numpy as np
import numpy.linalg as lin
from scipy import sparse as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


class Model:

    NUM_OF_IMAGES = 40

    HEIGHT = 512
    WEIGHT = 612

    NUM_OF_PIXELS = 512*612

    def __init__(self, obj_name):
        """
        obj_name: name of object from data
        M: observation matrix
        E: light intensities
        L : light directions
        Bt : surface normal transposed
        mask : mask for boundary of images
        """

        self.obj_name = obj_name

        self.M = load_data.load_observations(self.obj_name, self.NUM_OF_IMAGES)
        self.E = None
        self.L = load_data.load_lights(self.obj_name, self.NUM_OF_IMAGES)
        self.Bt = None

        self.normal_gt = load_data.load_normal_gt(self.obj_name)
        self.mask = cv2.imread('dataset/'+self.obj_name+'mask.png', cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        self.mask = np.where(self.mask == 255, 1, 0)

    def linear_joint(self):
        method = 'linear joint'
        print('***** Linear Joint Estimation method *****')

        # init
        self.E = None
        self.Bt = None

        Ip = sp.identity(self.NUM_OF_PIXELS)

        left_of_d = sp.kron(Ip, self.L)
        # right_of_d = np.diag(self.M[:, 0])
        right_of_d = sp.diags(self.M[:, 0])

        for pix in range(1,self.NUM_OF_PIXELS):
            # right_of_d = np.vstack((right_of_d, np.diag(self.M[:, pix])))
            right_of_d = sp.vstack([right_of_d, sp.diags(self.M[:, pix])])

            if pix % 100000 == 0:  # check
                print(pix)

        right_of_d = np.array(right_of_d).T
        print(right_of_d.shape)

        d = sp.hstack([left_of_d, right_of_d])
        u, s, vt = svds(d, k=1, which='SM')
        print(s)

        # equation = d.dot(vt.T)
        y = vt[-1:, 3*self.NUM_OF_PIXELS]
        e = vt[-1:, 3*self.NUM_OF_PIXELS:]

        self.E = e
        self.Bt = utils.normalize(y)

        results = np.reshape(self.Bt, (100, 100, -1))
        results = results*self.mask
        utils.plot_normal(results, method)
        return

    def factorization(self):
        method = 'factorization'
        print('***** Factorization based method *****')
        # init
        self.E = None
        self.Bt = None
        # step1
        U, S, Bt = np.linalg.svd(self.M, full_matrices=False)
        # print(S)
        root_S = np.diag(np.sqrt(S[:3]))
        Shat = U[:, :3].dot(root_S)
        Bt = root_S.dot(Bt[:3, :])

        D = np.zeros((2 * self.NUM_OF_IMAGES, 9))

        for i in range(self.NUM_OF_IMAGES):
            s = Shat[i]

            D[i * 2, :3] = [0, 0, 0]
            D[i * 2, 3:6] = -self.L[i, 2] * s
            D[i * 2, 6:] = self.L[i, 1] * s

            D[i * 2 + 1, :3] = self.L[i, 2] * s
            D[i * 2 + 1, 3:6] = [0, 0, 0]
            D[i * 2 + 1, 6:] = -self.L[i, 0] * s

        D = ((D-D.min())/(D.max()-D.min()))*2-1
        # step2
        U, S, Vt = np.linalg.svd(D, full_matrices=False)

        H = Vt[-1, :].reshape(-1, 3).T
        Bt = np.linalg.inv(H).dot(Bt).T

        # homogeneous equation
        # temp = D.dot(Vt[-1, :].T)

        self.Bt = (Bt-Bt.min())/(Bt.max()-Bt.min())  # normalize
        results = np.reshape(self.Bt, (self.HEIGHT, self.WEIGHT, -1))
        results = results*self.mask  # *self.mask

        utils.angular_error(self.normal_gt, self.Bt*2-1)
        utils.plot_normal(results, method)
        return

    def alternating_minimization(self):
        method = 'alternating minimization'
        print('***** Alternating Minimization method *****')

        #init
        self.E = None
        self.Bt = None

        E = np.identity(self.NUM_OF_IMAGES)
        Bt = np.zeros((3, self.NUM_OF_PIXELS))

        n_iters = 30

        for t in range(n_iters):
            # update B
            Bt, _, _, _ = np.linalg.lstsq(E.dot(self.L), self.M, rcond=None)
            next_E = np.zeros(self.NUM_OF_IMAGES)

            # update E
            for i in range(self.NUM_OF_IMAGES):
                numerator = np.sum(self.L[i, :].T * (self.M[i, :].dot(Bt.T)))
                denominator = np.sum(np.square(self.L[i, :].dot(Bt)))
                next_E[i] = numerator / denominator
                # original
                # for j in range(self.NUM_OF_PIXELS):
                #    numerator += (self.M[i, j]*self.L[i, :]).dot(Bt[:, j])
                #    denominator += (self.L[i, :].dot(Bt[:, j]))**2

            next_E = next_E / np.linalg.norm(next_E)
            E = np.diag(next_E)

            self.E = E
            self.Bt = Bt

        self.Bt = utils.normalize(self.Bt.T)  # normalization

        results = np.reshape(self.Bt, (self.HEIGHT, self.WEIGHT, -1))
        results = results*self.mask

        utils.angular_error(self.normal_gt, self.Bt)
        utils.plot_normal(results, method)
        return
