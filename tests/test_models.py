import numpy as np

import cubepy as cp

# def f(x, evt_idx, v):
#     print(x.shape, evt_idx.shape, v.shape, sep="\n", end="\n\n")
#     return np.sin(x) * v[:, None, evt_idx]


# a = np.zeros((3, 10))
# b = np.ones((3, 10))
# c = np.ones((3, 10), dtype=float)

# print(cp.integrate(f, a, b, args=(c,), evt_idx_arg=True, abstol=1e-8, reltol=1e-8))
def us_std_atm_density(z, earth_radius=6371):
    H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852])
    Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    # fmt: off
    P_b = 1.01325e5 * np.array(
        [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4,
         3.9046834e-5, 3.68501e-6, ])
    # fmt: on

    Rstar = 8.31432e3
    M0 = 28.9644
    gmr = 34.163195

    z = np.asarray(z)

    h = z * earth_radius / (z + earth_radius)
    i = np.searchsorted(H_b, h, side="right") - 1

    deltah = h - H_b[i]

    temperature = T_b[i] + Lm_b[i] * deltah

    mask = Lm_b[i] == 0
    pressure = np.full(z.shape, P_b[i])
    pressure[mask] *= np.exp(-gmr * deltah[mask] / T_b[i][mask])
    pressure[~mask] *= (T_b[i][~mask] / temperature[~mask]) ** (gmr / Lm_b[i][~mask])

    density = (pressure * M0) / (Rstar * temperature)  # kg/m^3
    return 1e-3 * density  # g/cm^3


def slant_depth_integrand(z, evt_idx, theta_tr, earth_radius, rho=us_std_atm_density):

    i = earth_radius ** 2 * np.cos(theta_tr[evt_idx]) ** 2
    j = z ** 2
    k = 2 * z * earth_radius

    ijk = i + j + k

    return 1e5 * rho(z, earth_radius) * ((z + earth_radius) / np.sqrt(ijk))


def slant_depth(
    z_lo,
    z_hi,
    theta_tr,
    earth_radius=6378.1,
    epsabs=1e-2,
    epsrel=1e-2,
):
    """
    Slant-depth in g/cm^2 from equation (3) in https://arxiv.org/pdf/2011.09869.pdf

    Parameters
    ----------
    z_lo : float
        Starting altitude for slant depth track.
    z_hi : float
        Stopping altitude for slant depth track.
    theta_tr: float, array_like
        Trajectory angle in radians between the track and earth zenith.
    earth_radius: float
        Radius of a spherical earth. Default from nuspacesim.constants
    func: callable
        The integrand for slant_depth. If None, defaults to `slant_depth_integrand()`.

    Returns
    -------
    x_sd: ndarray
        slant_depth g/cm^2
    err: (float) numerical error.

    """

    theta_tr = np.asarray(theta_tr)

    # def f(x, evt_idx, theta_tr):
    #     return func(x, theta_tr=theta_tr, earth_radius=earth_radius)

    return cp.integrate(
        slant_depth_integrand,
        z_lo,
        z_hi,
        args=(theta_tr, earth_radius),
        is_1d=True,
        evt_idx_arg=True,
        abstol=1e-8,
        reltol=1e-8,
        tile_byte_limit=2 ** 25,
        parallel=True,
    )


if __name__ == "__main__":
    N = int(1e7)
    a = np.zeros(N)
    b = np.ones(N)
    t = np.linspace(0.0, 40.0, N)

    v, e = slant_depth(a, b, t)
    # print(v.shape, e.shape)
    print(v, e)
