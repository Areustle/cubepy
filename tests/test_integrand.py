import numpy as np
from timeit import timeit

if __name__ == '__main__':

    N=int(1e8)
    a = np.ones((4,N,4))

    print(timeit(lambda: np.prod(a, 0), number=1))
    print(timeit(lambda: np.prod(a, -1), number=1))
