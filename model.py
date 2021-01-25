import load_data
import utils
import numpy as np
import numpy.linalg as lin
from sklearn.preprocessing import normalize


class Model:

    NUM_OF_IMAGES = 50

    HEIGHT = 512
    WEIGHT = 612

    NUM_OF_PIXELS = HEIGHT*WEIGHT

    def __init__(self, obj_name):
        """
        M : f*p
        E : f*f
        L : f*3
        Bt : 3*p
        mask : H*W
        normal_gt : p*3
        """

        self.obj_name = obj_name

        self.M = load_data.load_observations(self.obj_name, self.NUM_OF_IMAGES)
        self.E = None
        self.L = load_data.load_lights(self.obj_name, self.NUM_OF_IMAGES)
        self.Bt = None

        self.mask = load_data.load_mask(self.obj_name)
        self.normal_gt = load_data.load_normal_gt(self.obj_name)

    def factorization(self):
        method = 'factorization'
        print('***** Factorization based method *****')
        # init
        self.E = None
        self.Bt = None

        U, S, Bt = np.linalg.svd(self.M, full_matrices=False)

        # linear ambiguity between Shat and Bt
        root_S = np.diag(np.sqrt(S[:3]))
        Shat = U[:, :3].dot(root_S)
        Bt = root_S.dot(Bt[:3, :])

        # to remove linear ambiguity
        D = np.zeros((2 * self.NUM_OF_IMAGES, 9))

        for i in range(self.NUM_OF_IMAGES):
            D[i * 2, :3] = [0, 0, 0]
            D[i * 2, 3:6] = -self.L[i, 2] * Shat[i]
            D[i * 2, 6:] = self.L[i, 1] * Shat[i]

            D[i * 2 + 1, :3] = self.L[i, 2] * Shat[i]
            D[i * 2 + 1, 3:6] = [0, 0, 0]
            D[i * 2 + 1, 6:] = -self.L[i, 0] * Shat[i]

        # to solve homogeneous equation
        U, S, Vt = np.linalg.svd(D, full_matrices=False)

        H = Vt[-1, :].reshape((3, 3)).T
        E = Shat.dot(H)
        Bt = np.linalg.inv(H).dot(Bt)

        self.Bt = normalize(Bt, axis=0)
        self.E = normalize(E, axis=0)

        angular_error_map = utils.angular_error(self.normal_gt, self.Bt.T, self.mask.ravel())
        angular_error_map = angular_error_map.reshape((self.HEIGHT, self.WEIGHT, -1))

        results = np.reshape((self.Bt.T+1)/2.0, (self.HEIGHT, self.WEIGHT, -1))
        results = results*self.mask[:, :, np.newaxis]

        utils.plot_normal(results, angular_error_map, method)
        return

    def alternating_minimization(self):
        method = 'alternating minimization'
        print('***** Alternating Minimization method *****')

        #init
        self.E = None
        self.Bt = None

        E = np.identity(self.NUM_OF_IMAGES)
        Bt = None

        n_iters = 50  # 30~70

        for t in range(n_iters):
            # update B
            Bt = np.linalg.lstsq(E.dot(self.L), self.M, rcond=None)[0]
            next_E = np.zeros(self.NUM_OF_IMAGES)
            # update E
            for i in range(self.NUM_OF_IMAGES):
                numerator = np.sum(self.L[i, :].T * (self.M[i, :].dot(Bt.T)))
                denominator = np.sum((self.L[i, :].dot(Bt))**2)
                next_E[i] = numerator / denominator

            next_E = next_E / np.linalg.norm(next_E)  # normalization E, to avoid the oscillation
            E = np.diag(next_E)

        self.E = E
        self.Bt = Bt

        self.Bt = normalize(self.Bt, axis=0).T  # normalization
        results = np.reshape(self.Bt, (self.HEIGHT, self.WEIGHT, -1))
        results = (results+1)/2.0 * self.mask[:, :, np.newaxis]

        error_map = utils.angular_error(self.normal_gt, self.Bt, self.mask.ravel())
        error_map = error_map.reshape((self.HEIGHT, self.WEIGHT, -1))*self.mask[:, :, np.newaxis]
        utils.plot_normal(results, error_map, method)

        return
