import jax
import jax.numpy as jnp
from scipy.special import comb, factorial

def make_knots(t_mesh, k):
    """Return knot vector with k+1 ghost knots on each side."""
    r = k + 1
    t_mesh = jnp.asarray(t_mesh, float)

    dt_left  = t_mesh[1] - t_mesh[0]
    dt_right = t_mesh[-1] - t_mesh[-2]

    left  = t_mesh[0] - dt_left  * jnp.arange(r, 0, -1)
    right = t_mesh[-1] + dt_right * jnp.arange(1, r+1)

    return jnp.concatenate([left, t_mesh, right])  # shape: (N + 2*(k+1))

def bases(t, knots, k, normalize=True):
    """
    Compute B-spline basis of degree k on knot vector.
    Return array of shape (num_basis, len(t)).
    """
    t = jnp.atleast_1d(t)
    K = len(knots)
    r = k + 1       # order of finite difference
    p = k           # relu power = k

    # Differences coefficients (length r+1)
    coeffs = jnp.array([(-1)**j * comb(r, j) for j in range(r+1)], float)

    # relu^k evaluated on each knot
    Y = jnp.maximum(0.0, t[None, :] - knots[:, None])**p   # shape (K, M)

    m = K - r          # number of splines
    phi = jnp.zeros((m, t.shape[0]))

    for j in range(r+1):
        phi = phi + coeffs[j] * Y[j:j+m, :]

    if normalize:
        phi = phi / factorial(k)

    return phi  # shape (num_basis, len(t))

def model(params, t, knots, k):
    """
    params: shape (num_basis,)
    t: scalar or vector
    """
    B = bases(t, knots, k)
    return jnp.dot(params, B)   # shape: (len(t),)



