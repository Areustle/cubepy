from __future__ import annotations

from typing import Callable

import numpy as np

from .type_aliases import NPF, NPI


def gauss_kronrod(
    f: Callable,
    pts: NPF,
    halfwidths: NPF,
    *_,
):
    # GK [7, 15] weights from
    # https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
    gk_weights = np.array(
        [
            0.022935322010529224963732008058970,  # 0
            0.063092092629978553290700663189204,  # 1
            0.104790010322250183839876322541518,  # 2
            0.140653259715525918745189590510238,  # 3
            0.169004726639267902826583426598550,  # 4
            0.190350578064785409913256402421014,  # 5
            0.204432940075298892414161999234649,  # 6
            0.209482141084727828012999174891714,  # 7
            0.204432940075298892414161999234649,  # 6
            0.190350578064785409913256402421014,  # 5
            0.169004726639267902826583426598550,  # 4
            0.140653259715525918745189590510238,  # 3
            0.104790010322250183839876322541518,  # 2
            0.063092092629978553290700663189204,  # 1
            0.022935322010529224963732008058970,  # 0
        ]
    )

    gl_weights = np.array(
        [
            0.129484966168869693270611432679082,  # 0
            0.279705391489276667901467771423780,  # 1
            0.381830050505118944950369775488975,  # 2
            0.417959183673469387755102040816327,  # 3
            0.381830050505118944950369775488975,  # 2
            0.279705391489276667901467771423780,  # 1
            0.129484966168869693270611432679082,  # 0
        ]
    )

    # centers, halfwidths  =  1 * [ regions, { 1 | nevts } ]
    # p.shape 1 * [ 15(points), regions, { 1 | nevts } ]
    # pts = points.gk_pts(center[0], halfwidths[0])
    nreg = pts[0].shape[1]
    nevt = pts[0].shape[2]
    pts = pts.reshape(1, 15 * nreg, nevt)
    # vals.shape [ points, regions, events ]
    vals = f(pts).reshape(15, nreg, nevt)

    # Reshape shapes to conform to matmul shape requirements.
    # s0 = (15, nreg, nevt)
    # s1 = (15, nreg * nevt)
    # s2 = (2, nreg, nevt)

    # r_ [ regions, events ] = [M] . [ M, regions, events ]
    rg = np.tensordot(gl_weights, vals[1::2], (0, 0))
    rk = np.tensordot(gk_weights, vals, (0, 0))

    # error [ regions, events ]
    err = halfwidths[0] * np.abs(rk - rg)

    mean = 0.5 * rk
    I_tilde = halfwidths[0] * np.tensordot(gk_weights, np.abs(vals - mean), (0, 0))

    mask = np.abs(I_tilde) > 1.0e-15
    scale = (200.0 * err[mask] / I_tilde[mask]) ** 1.5
    scale[scale > 1.0] = 1.0
    err[mask] = I_tilde[mask] * scale

    min_err = 50.0 * np.finfo(rk.dtype).eps
    rabs = halfwidths[0] * np.tensordot(gk_weights, np.abs(vals), (0, 0))
    err[(rabs > (np.finfo(rk.dtype).tiny / min_err)) & (min_err > err)] = min_err

    return (rk * halfwidths[0]), err, np.zeros(err.shape[0], dtype=int)
