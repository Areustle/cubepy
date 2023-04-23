import numpy as np

from cubepy import input


def test_input_1():
    lo, hi, es = input.parse_input(lambda x: x, 0.0, 1.0, [])
    assert lo == np.array([0.0])
    assert hi == np.array([1.0])
    assert es == ()


def test_input_2():
    lo, hi, es = input.parse_input(lambda x, a: a + x, 0.0, 1.0, [1])
    assert lo == np.array([0.0])
    assert hi == np.array([1.0])
    assert es == ()


def test_input_3():
    lo, hi, es = input.parse_input(
        lambda x, a, b: a + b * x, 0.0, 1.0, [1, np.arange(8)]
    )
    assert lo == np.array([0.0])
    assert hi == np.array([1.0])
    assert es == (8,)


def test_input_4():
    lo, hi, es = input.parse_input(
        lambda x, y, a, b: a + b * x * y, np.zeros(2), np.ones(2), [1, np.arange(8)]
    )
    assert np.all(lo == np.zeros(2))
    assert np.all(hi == np.ones(2))
    assert es == (8,)


def test_input_5():
    lo, hi, es = input.parse_input(
        lambda x, y, *, a, b: a + b * x * y, np.zeros(2), np.ones(2), [1, np.arange(8)]
    )
    assert np.all(lo == np.zeros(2))
    assert np.all(hi == np.ones(2))
    assert es == (8,)


def test_input_6():
    lo, hi, es = input.parse_input(
        lambda x, y, *, a, b: a + b * x * y,
        np.zeros(2),
        np.ones(2),
        [np.arange(8), np.arange(8)],
    )
    assert np.all(lo == np.zeros(2))
    assert np.all(hi == np.ones(2))
    assert es == (8,)
