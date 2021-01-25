from model import Model


if __name__ == '__main__':

    # select one object
    # OBJ = 'bearPNG/'
    OBJ = 'catPNG/'
    # OBJ = 'readingPNG/'

    model = Model(OBJ)

    # model.factorization()
    model.alternating_minimization()
