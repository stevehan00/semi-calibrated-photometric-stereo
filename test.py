from model import Model


if __name__ == '__main__':
    OBJ = 'bearPNG/'
    # OBJ = 'catPNG/'
    # OBJ = 'ballPNG/'
    # OBJ = 'harvestPNG/'

    model = Model(OBJ)
    # model.linear_joint()
    # model.factorization()
    # model.alternating_minimization()
    model.am_solution()