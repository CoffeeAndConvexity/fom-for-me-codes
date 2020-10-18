import numpy as np
from numba import jit


def sum_squares(yy):
    return np.sum(yy**2)

def proj_simplex(v, s = 1, return_multiplier=False): # assuming v is np.array
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    ind = np.arange(1, n+1)
    cond = u - cssv / ind > 0
    theta = cssv[cond][-1] / float(ind[cond][-1]) # rho = float(ind[cond][-1])
    w = np.maximum(v - theta, 0)
    if not return_multiplier:
        return w
    # if return multiplier, i = np.argmax(w > 0)
    return w, theta

def proj_simplex_all(x, s=None, return_multipliers=False):
    n, m = x.shape # project each x[:, j] onto s[j]-simplexes
    s = (np.ones((m,)) if s is None else s) # len(s) == m
    u = np.sort(x, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - s # cssv[:, j] = np.cumsum(u[:, j]) - s[j]
    ind = np.arange(1, n+1)
    cond = u - (cssv.T/ind).T > 0
    # theta_vec = [cssv[:, j][cond[:,j]][-1] for j in range(m)] / np.array([ind[cond[:,j]][-1] for j in range(m)], dtype=np.double)
    pivot_indices = np.argmin(cond, axis=0) - 1
    pivot_indices[pivot_indices==-1] = n-1
    theta_vec = cssv[pivot_indices, range(m)] / ind[pivot_indices]
    w = np.maximum(x - theta_vec, 0)
    if not return_multipliers:
        return w
    return w, theta_vec

def unit_proj_simplex_all(x, return_multipliers=False):
    n, m = x.shape # project each x[:, j] onto s[j]-simplexes
    u = np.sort(x, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1 # cssv[:, j] = np.cumsum(u[:, j]) - s[j]
    ind = np.arange(1, n+1)
    cond = u - (cssv.T/ind).T > 0
    # theta_vec = [cssv[:, j][cond[:,j]][-1] for j in range(m)] / np.array([ind[cond[:,j]][-1] for j in range(m)], dtype=np.double)
    pivot_indices = np.argmin(cond, axis=0) - 1
    # pivot_indices[pivot_indices==-1] = n-1
    theta_vec = cssv[pivot_indices, range(m)] / ind[pivot_indices]
    w = np.maximum(x - theta_vec, 0)
    if not return_multipliers:
        return w
    return w, theta_vec

def weighted_proj_simplex(x, w, t = 1, return_obj = False):
    """ x: vector to be projected
        w > 0: diagonal elements of the weight matrix
        t > 0: scale of the simplex (x>=0, sum(x) == t) """
    # easy cases
    if len(x) == 1: # trivial case
        return t
    if w is None:
        return proj_simplex(x, t)
    # convert all to array
    x, w, n = np.array(x), np.array(w), len(x)
    # use the O(n log n algorithm)
    xtil = x * w
    sorted_indices = np.argsort(-xtil) # sort x * w from in decreasing order
    lvals = -xtil[sorted_indices] # -x[0] <= -x[1] <= ... <= -x[n-1]
    wp = w[sorted_indices] # w permuted according to the sorting of x * w
    L = M = 0 # find largest k such that S[k] = [sum of 1/wp[i] * (lvals[k] - lvals[i]) over i = 0, ..., k] <= 1
    for k in range(n):
        L += 1 / wp[k]
        M += lvals[k] / wp[k]
        if k == n -1 or L * lvals[k+1] - M >= t:
            break
    # now S[k] < 1 and S[k+1] >= 1, therefore lam is between lvals[k] and lvals[k+1]
    lam = (t + M) / L # for debugging: phi = lambda lll: np.sum(np.maximum(lll + x * sig, 0)/w); phi(lam)
    y = np.maximum(x+lam/w, 0) # find y
    if not return_obj:
        return y
    aa = y - x
    return y, 0.5 * (aa.dot(w*aa))

# def vertex_to_mat(vertex): # convert vertex to a matrix in FW
#     xx = np.zeros((n, m))
#     xx[vertex, range(m)] = 1
#     return xx