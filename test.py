from model import Model


if __name__ == '__main__':

    # select one object
    # OBJ = 'bearPNG/'
    # OBJ = 'catPNG/'
    OBJ = 'readingPNG/'
    # OBJ = 'harvestPNG/'

    model = Model(OBJ)
    # model.linear_joint()
    model.factorization()
    model.alternating_minimization()
