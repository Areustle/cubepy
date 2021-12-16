import numpy as np

from cubepy import region


def test_region():

    lo = np.zeros(1)
    hi = np.ones(1)

    r = region.region(lo, hi)

    print(r)


test_region()
