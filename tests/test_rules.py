import numpy as np
from cubepy.rules import genz_malik


# def test_genz_malik():

if __name__ == '__main__':


    # def I1(x):
    #     return x

    # c = np.zeros((1, 1, 1))
    # h = np.ones((1, 1, 1))
    # v = np.ones((1, 1))

    # r, e, s = genz_malik(I1, c, h, v)
    # print(c.shape, r.shape, e.shape, s.shape)

    # c = np.zeros((1, 5, 10))
    # h = np.ones((1, 5, 10))
    # v = np.ones((5, 10))

    # r, e, s = genz_malik(I1, c, h, v)
    # print(c.shape, r.shape, e.shape, s.shape)


    def fp(x):
        return np.prod(x, 0)

    # c = np.zeros((2, 10, 1))
    # h = np.ones((2, 10, 1))
    # v = np.ones((10, 1))

    # r, e, s = genz_malik(fp, c, h, v)
    # print(c.shape, r.shape, e.shape, s.shape)

    # c = np.zeros((3, 10, 1))
    # h = np.ones((3, 10, 1))
    # v = np.ones((10, 1))

    # r, e, s = genz_malik(fp, c, h, v)
    # print(c.shape, r.shape, e.shape, s.shape)

    regions = 60
    events = 1000000

    c = np.zeros((2, regions, events))
    h = np.ones((2, regions, events))
    v = np.ones((regions, events))

    r, e, s = genz_malik(fp, c, h, v)
    print(c.shape, r.shape, e.shape, s.shape)
