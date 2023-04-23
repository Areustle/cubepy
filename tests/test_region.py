import numpy as np

from cubepy import region


def test_region_simple():
    def tr(lo, hi):
        c, h, v = region.region(lo, hi)
        for ci in c:
            assert np.all(ci == 0.5)
        for hi in h:
            assert np.all(hi == 0.5)
        for vi in v:
            assert np.all(vi == 1.0)

    tr(np.zeros(1), np.ones(1))
    tr(np.zeros((1)), np.ones((1)))
    tr(np.zeros((1, 1)), np.ones((1, 1)))
    tr(np.zeros((1, 1024)), np.ones((1, 1024)))
    tr(np.zeros((2, 1024)), np.ones((2, 1024)))
    tr(np.zeros((3, 1024)), np.ones((3, 1024)))
    tr(np.zeros((4, 1024)), np.ones((4, 1024)))
    tr(np.zeros((5, 1024)), np.ones((5, 1024)))
    tr(np.zeros((6, 1024)), np.ones((6, 1024)))
    tr(np.zeros((7, 1024)), np.ones((7, 1024)))
    tr(np.zeros((8, 1024)), np.ones((8, 1024)))


def test_region_complex_1d():
    lo = np.expand_dims(np.arange(1.0, 1e5), 0)
    lo -= 0.5e5
    hi = lo + 1.0

    c, h, v = region.region(lo, hi)

    assert len(c) == 1
    assert len(h) == 1
    assert type(v) == np.ndarray

    assert np.all(c[0] == lo + 0.5)
    assert np.all(h[0] == 0.5)
    assert np.all(v == 1.0)


def test_split_1():
    c = np.array([[0.5]])
    h = np.array([[0.5]])
    v = np.array([1.0])
    s = np.array([0])
    c, h, v = region.split(c, h, v, s, np.array([True]))

    assert np.all(c == [0.25, 0.75])
    assert np.all(h == [0.25, 0.25])
    assert np.all(v == [0.5, 0.5])


def test_split_2():
    low = np.array([0.0, 0.0])
    high = np.array([1.0, np.pi])

    c, h, v = region.region(low, high)
    assert np.all(c == [[0.5], [0.5 * np.pi]])
    assert np.all(h == [[0.5], [0.5 * np.pi]])
    assert np.all(v == [np.pi])

    s = np.array([1])
    umask = np.array([True])
    c1, h1, v1 = region.split(c, h, v, s, umask)
    assert np.all(c1 == [[0.5, 0.5], [0.25 * np.pi, 0.75 * np.pi]])
    assert np.all(h1 == [[0.5, 0.5], [0.25 * np.pi, 0.25 * np.pi]])
    assert np.all(v1 == [0.5 * np.pi, 0.5 * np.pi])

    s = np.array([0])
    umask = np.array([True])
    c1, h1, v1 = region.split(c, h, v, s, umask)
    assert np.all(c1 == [[0.25, 0.75], [0.5 * np.pi, 0.5 * np.pi]])
    assert np.all(h1 == [[0.25, 0.25], [0.5 * np.pi, 0.5 * np.pi]])
    assert np.all(v1 == [0.5 * np.pi, 0.5 * np.pi])

    s = np.array([0, 0])
    umask = np.array([True, True])
    c2, h2, v2 = region.split(c1, h1, v1, s, umask)
    assert np.all(
        c2
        == [
            [0.125, 0.625, 0.375, 0.875],
            [
                0.5 * np.pi,
            ]
            * 4,
        ]
    )
    assert np.all(
        h2
        == [
            [
                0.125,
            ]
            * 4,
            [
                0.5 * np.pi,
            ]
            * 4,
        ]
    )
    assert np.all(
        v2
        == [
            0.25 * np.pi,
        ]
        * 4
    )

    # s = np.array([1, 1])
    # nmask = np.array([[True], [True]])
    # c2, h2, v2 = region.split(c1, h1, v1, s, nmask)
    # assert np.all(
    #     c2
    #     == [
    #         [0.125, 0.625, 0.375, 0.875],
    #         [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi],
    #     ]
    # )
    # assert np.all(
    #     h2
    #     == [
    #         [0.125, 0.125, 0.125, 0.125],
    #         [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi],
    #     ]
    # )
    # assert np.all(v2 == [0.25 * np.pi, 0.25 * np.pi, 0.25 * np.pi, 0.25 * np.pi])


# def test_region_complex_2d():
#     lo = np.arange(1.0, 1e5)
#     lo -= 0.5e5
#     lo = np.stack((lo, lo), 0)
#
#     assert lo.ndim == 2
#     assert lo.shape[0] == 2
#
#     hi = lo + 1.0
#
#     c_true = lo + 0.5
#     h_true = 0.5
#     v_true = 1.0
#
#     c, h, v = region.region(lo, hi)
#
#     assert len(c) == 2
#     assert len(h) == 2
#     assert v.shape[0] == lo.shape[1]
#
#     assert np.all(c[0] == c_true)
#     assert np.all(c[1] == c_true)
#     assert np.all(h[0] == h_true)
#     assert np.all(h[1] == h_true)
#     assert np.all(v == v_true)
#
#
# def test_region_random_4d():
#     lo = np.random.uniform(low=-1e4, high=1e4, size=(4, int(1e5)))
#
#     assert lo.ndim == 2
#     assert lo.shape[0] == 4
#
#     hi = lo + 1.0
#
#     c_true = lo + 0.5
#     h_true = 0.5
#     v_true = 1.0
#
#     c, h, v = region.region((*lo,), (*hi,))
#
#     assert len(c) == 4
#
#     for i, ci in enumerate(c):
#         assert np.all(ci == c_true[i])
#     for hi in h:
#         assert np.all(hi == h_true)
#     assert np.all(v == v_true)

# def test_split():
#     c0 = np.ones((1, 1, 10)) * 0.5
#     h0 = np.ones((1, 1, 10)) * 0.5
#     v0 = np.ones((1, 1, 10))
#     sd = np.zeros((1, 10), dtype=int)
#
#     c1, h1, v1 = region.split(*c0, *h0, *v0, *sd)
#
#     assert c1.ndim == c0.ndim
#     assert h1.ndim == h0.ndim
#     assert v1.ndim == v0.ndim
#
#     assert np.all(c1[:, : (c1.shape[1] // 2), ...] == 0.25), f"{c1}"
#     assert np.all(c1[:, (c1.shape[1] // 2) : -1, ...] == 0.75)
#     assert np.all(h1 == 0.25)
#     assert np.all(v1 == 0.5)

# def test_split_complex():

#     lo = np.arange(1.0, 5)
#     lo -= 2.5
#     lo = np.stack((lo, lo), 0)

#     assert lo.ndim == 2
#     assert lo.shape[0] == 2

#     hi = lo + 1.0

#     c0, h0, v0 = region.region(lo, hi)

#     assert np.all(c0.shape == (2, 4))

#     c_ = np.copy(c0)
#     h_ = np.copy(h0)
#     v_ = np.copy(v0)

#     sd = np.zeros(4, dtype=int)
#     sd[1::2] = 1

#     c1, h1, v1 = region.split(c0, h0, v0, sd)

#     assert c1.ndim == c_.ndim
#     assert h1.ndim == h0.ndim
#     assert v1.ndim == v0.ndim

#     assert c1.shape[0] == c_.shape[0]
#     assert c1.shape[1] == c_.shape[1] * 2

#     assert h1.shape[0] == h_.shape[0]
#     assert h1.shape[1] == h_.shape[1] * 2

#     assert v1.shape[0] == v_.shape[0] * 2

#     for i in range(c_.shape[1]):

#         j = i + c_.shape[1]

#         assert np.all(
#             c_[0, i] - 0.25 == c1[0, i]
#         ), f"{i} {j} \n {c_.squeeze()} \n {c1.squeeze()}"

#         assert np.all(
#             c_[0, i] + 0.25 == c1[0, j]
#         ), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"

#         assert np.all(c_[1, i] == c1[1, i]), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"

#         assert np.all(c_[1, i] == c1[1, j]), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"
