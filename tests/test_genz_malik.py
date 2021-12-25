import numpy as np

import cubepy as cp

if __name__ == "__main__":

    def integrand(x):
        return 1 + 8 * x[0] * x[1]

    def exact(r):
        return np.prod(r, axis=0)

    low = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    high = np.array(
        [
            [3.0],
            [2.0],
        ]
    )

    print(low.shape, low)

    center, hwidth, vol = cp.region.region(low, high)

    print("center", center, center.shape)
    print("hwidth", hwidth, hwidth.shape)
    print("vol", vol, vol.shape)

    value, error, split_dim = cp.genz_malik.genz_malik(integrand, center, hwidth, vol)

    print("value", np.squeeze(value), value.shape)
    print("error", np.squeeze(error), error.shape)
    print("split_dim", np.squeeze(split_dim), split_dim.shape)
