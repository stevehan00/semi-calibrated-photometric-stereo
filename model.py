import load_data
import cv2
import numpy as np
import numpy.linalg as lin
from scipy import sparse as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


class Model:

    NUM_OF_IMAGES = 96

    HEIGHT = 512
    WEIGHT = 612

    NUM_OF_PIXELS = 512*612*3

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

        self.M = load_data.load_observations(self.obj_name)
        self.E = None
        self.L = load_data.load_lights(self.obj_name)
        self.Bt = None

        self.mask = cv2.imread('dataset/'+self.obj_name+'mask.png', cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    def linear_joint(self):
        print('***** Linear Joint Estimation method *****')

        # init
        self.E = None
        self.Bt = None

        # step1
        Ip = sp.identity(self.NUM_OF_IMAGES)

        left_of_d = sp.kron(Ip, self.L)
        right_of_d = np.diag(self.M[:, 0])

        for pix in range(1, self.NUM_OF_PIXELS):
            right_of_d = np.vstack((right_of_d, np.diag(self.M[:, pix])))
            print(pix)

        d = sp.hstack([left_of_d, right_of_d])
        print('completed step1!')
        # step2
        u, s, vt = svds(d, k=1, which='SM')
        print(s)  # #

        equation = d.dot(vt.T)
        y = vt[-1:, 3*self.NUM_OF_PIXELS]
        y = ((y - y.min())/(y.max()-y.min()))*2-1
        self.Bt = np.reshape(y, (100, 100, -1))

        print('completed step2!')

        results = ((0.5*(self.Bt+1))*255)*self.mask

        print(results.shape)

    def factorization(self):
        print('***** Factorization based method *****')
        # init
        self.E = None
        self.Bt = None

        # step1
        U, S, Bt = np.linalg.svd(self.M, full_matrices=False)

        root_S = np.diag(np.sqrt(S[:3]))

        Shat = U[:, :3].dot(root_S)
        Bt = root_S.dot(Bt[:3, :])

        # print(U.shape, S.shape, Bt.shape)
        # print(Shat.shape, Bt.shape)

        D = np.zeros((2 * self.NUM_OF_IMAGES, 9))

        for i in range(self.NUM_OF_IMAGES):
            s = Shat[i]

            D[i * 2, :3] = [0, 0, 0]
            D[i * 2, 3:6] = -self.L[i, 2] * s
            D[i * 2, 6:] = self.L[i, 1] * s

            D[i * 2 + 1, :3] = self.L[i, 2] * s
            D[i * 2 + 1, 3:6] = [0, 0, 0]
            D[i * 2 + 1, 6:] = -self.L[i, 0] * s
        print('completed step1 !')

        # step2
        # U, S, Vt = np.linalg.svd(D, full_matrices=False)
        U,S,Vt = sp.linalg.svds(D, which='SM', k=1)

        H = Vt[-1, :].reshape(-1, 3).T
        print(S)

        Bt = np.linalg.inv(H).dot(Bt).T

        temp = D.dot(Vt[-1, :].T)

        plt.title('D*y = 0')
        plt.plot(temp.ravel())
        plt.show()

        normalized_normal = ((Bt - Bt.min()) / (Bt.max() - Bt.min()))*2-1

        self.Bt = np.reshape(normalized_normal, (self.HEIGHT, self.WEIGHT, -1))

        results = ((0.5 * (self.Bt + 1)) * self.mask)  # *self.mask

        plt.plot(results.ravel())
        plt.show()

        gt = load_data.load_normal(self.obj_name)
        plt.plot(gt.ravel())
        plt.show()

        cv2.imshow('Factorization', gt)
        cv2.waitKey()
        cv2.destroyAllWindows()

        print(results.shape)

        cv2.imshow('Factorization', results)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def alternating_minimizing(self):
        print('***** Alternating Minimization method *****')

        #init
        self.E = None
        self.Bt = None

        E = np.identity(self.NUM_OF_IMAGES)
        Bt = np.zeros((3, self.NUM_OF_PIXELS))

        n_iters = 50

        for t in range(n_iters):
            next_B, _, _, _ = np.linalg.lstsq(E.dot(self.L), self.M, rcond=None)
            Bt = next_B
            next_E = np.zeros(self.NUM_OF_IMAGES)

            for i in range(self.NUM_OF_IMAGES):
                numerator = 0
                denominator = 0

                for j in range(self.NUM_OF_PIXELS):
                    numerator += (self.M[i, j]*self.L[i, :]).dot(Bt[:, j])
                    denominator += (self.L[i, :].dot(Bt[:, j]))**2
                next_E[i] = numerator/denominator

            next_E = next_E / np.linalg.norm(next_E)
            E = np.diag(next_E)

            self.E = E
            self.Bt = Bt

            print(t, 'of iters')
        print(self.E.shape, self.Bt.shape)

        B = self.Bt.T
        normalization_normal = ((B - B.min()) / (B.max() - B.min())) * 2 - 1  # -1 to 1 normalization

        results = np.reshape(normalization_normal, (100, 100, -1))
        results = ((0.5 * (results + 1)) * 255)*self.mask

        print(results.shape)