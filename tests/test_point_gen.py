import numpy as np
from cubepy import point_gen


def test_nums():
    n = np.arange(1, 10)

    assert np.all(point_gen.num_k0k1(n) == [5, 9, 13, 17, 21,  25,  29,  33,  37])
    assert np.all(point_gen.num_k2(n) == [0, 4, 12, 24, 40,  60,  84, 112, 144])
    assert np.all(point_gen.num_k6(n) == [2, 4,  8, 16, 32,  64, 128, 256, 512])
    assert np.all(point_gen.num_points(n) == [7, 17, 33, 57, 93, 149, 241, 401, 693])


def test_full_kn():

    for i in range(1, 10):

        c = np.ones((i))

        n01 = point_gen.num_k0k1(i)
        n2 = point_gen.num_k2(i)
        n6 = point_gen.num_k6(i)

        assert np.all(point_gen.full_kn(c, point_gen.num_k0k1) == np.ones((i, n01)))
        assert np.all(point_gen.full_kn(c, point_gen.num_k2) == np.ones((i, n2)))
        assert np.all(point_gen.full_kn(c, point_gen.num_k6) == np.ones((i, n6)))


def test_fulls():

    c = np.zeros((4))
    h  = np.ones((4))

    x01 = [[0, -1,  1, -2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
           [0,  0,  0,  0,  0, -1,  1, -2,  2,  0,  0,  0,  0,  0,  0,  0,  0,],
           [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1, -2,  2,  0,  0,  0,  0,],
           [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1, -2,  2,]]

    assert np.all(x01 == point_gen.pts_k0k1(c, h, 2*h))

    x2 = [[-2,  2, -2,  2, -2,  2, -2,  2, -2,  2, -2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
          [-2, -2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0, -2,  2, -2,  2, -2,  2, -2,  2,  0,  0,  0,  0,],
          [ 0,  0,  0,  0, -2, -2,  2,  2,  0,  0,  0,  0, -2, -2,  2,  2,  0,  0,  0,  0, -2,  2, -2,  2,],
          [ 0,  0,  0,  0,  0,  0,  0,  0, -2, -2,  2,  2,  0,  0,  0,  0, -2, -2,  2,  2, -2, -2,  2,  2,]]

    assert np.all(x2 == point_gen.pts_k2(c, 2*h))

    x6 = [[-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,],
          [-1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,],
          [-1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,],
          [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,]]

    assert np.all(x6 == point_gen.pts_k6(c, h))


def test_pts_k0k1():

    for i in range(1,10):

        n = point_gen.num_k0k1(i)

        c = np.zeros((i))
        h  = np.ones((i))
        assert point_gen.pts_k0k1(c, h, 2*h).shape == (i, n)

        c = np.zeros((i, 5))
        h  = np.ones((i, 5))
        assert point_gen.pts_k0k1(c, h, 2*h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h  = np.ones((i, 5, 6))
        assert point_gen.pts_k0k1(c, h, 2*h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h  = np.ones((i, 5, 6, 7))
        assert point_gen.pts_k0k1(c, h, 2*h).shape == (i, n, 5, 6, 7)


def test_pts_k2():

    for i in range(1,10):

        n = point_gen.num_k2(i)

        # [dim]
        c = np.zeros((i))
        h  = np.ones((i))
        assert point_gen.pts_k2(c, h).shape == tuple(filter(None, (i, n)))

        # [dim, regions]
        c = np.zeros((i, 5))
        h  = np.ones((i, 5))
        assert point_gen.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5)))

        # [dim, regions, events]
        c = np.zeros((i, 5, 6))
        h  = np.ones((i, 5, 6))
        assert point_gen.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5, 6)))

        # [dim, regions, events, xxx]
        c = np.zeros((i, 5, 6, 7))
        h  = np.ones((i, 5, 6, 7))
        assert point_gen.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5, 6, 7)))


def test_pts_k6():

    for i in range(1,10):

        n = point_gen.num_k6(i)

        c = np.zeros((i))
        h  = np.ones((i))
        assert point_gen.pts_k6(c, h).shape == (i, n)

        c = np.zeros((i, 5))
        h  = np.ones((i, 5))
        assert point_gen.pts_k6(c, h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h  = np.ones((i, 5, 6))
        assert point_gen.pts_k6(c, h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h  = np.ones((i, 5, 6, 7))
        assert point_gen.pts_k6(c, h).shape == (i, n, 5, 6, 7)


def test_fullsym():

    for i in range(1,10):

        n = point_gen.num_points(i)

        c = np.zeros((i))
        h  = np.ones((i))
        assert point_gen.fullsym(c, h, 2*h, h).shape == (i, n)

        c = np.zeros((i, 5))
        h  = np.ones((i, 5))
        assert point_gen.fullsym(c, h, 2*h, h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h  = np.ones((i, 5, 6))
        assert point_gen.fullsym(c, h, 2*h, h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h  = np.ones((i, 5, 6, 7))
        assert point_gen.fullsym(c, h, 2*h, h).shape == (i, n, 5, 6, 7)



# def test_fullsym():
#     v1 = np.ones((4))

#     v = np.concatenate(
#         (
#             point_gen.pts_k0k1(np.zeros((4)), v1, 2 * v1),
#             point_gen.pts_k2(np.zeros((4)), 2 * v1),
#             point_gen.pts_k6(np.zeros((4)), v1),
#         ),
#         axis=-1,
#     )

#     assert np.all(point_gen.fullsym(np.zeros((4, 1, 1)), v1, 2 * v1, v1) == v)


if __name__ == "__main__":
    np.set_printoptions(linewidth=256)
    # test_nums()
    # test_fulls()
    # test_fullsym()

    # [dim, regions, events]
    c = np.zeros((3,5,6))
    h  = np.ones((3,5,6))

    # [dim, points, regions, events]
    # print(point_gen.pts_k0k1(c, h, 2*h))
    print(point_gen.pts_k0k1(c, h, 2*h).shape)
    # print(point_gen.pts_k2(c, 2*h))
    print(point_gen.pts_k2(c, 2*h).shape)
    # print(point_gen.pts_k6(c, h))
    print(point_gen.pts_k6(c, h).shape)
    print(point_gen.fullsym(c, h, 2*h, h).shape)

