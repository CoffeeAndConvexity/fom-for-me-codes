#############################################################################
#############################################################################
#############################################################################
""" load module, define constants and functions"""

from utils import * 
import cvxpy as cp
from time import time
import os, csv, argparse
from collections import defaultdict
from scipy.optimize import bisect

np.random.seed(123)
accu = 1e-3
n, m = 50, 100
distr_name = 'unif'
print("n = {}, m = {}, distr = {}".format(n, m, distr_name))
max_iter = 2000

do_ls = True

# distribution for random matrix generation
if distr_name == "unif":
    distr = np.random.uniform # np.random.exponential
elif distr_name == "exp":
    distr = np.random.exponential
elif distr_name == 'normal': 
    distr = np.random.normal
elif distr_name == 'log_normal' or distr_name == 'lognormal':
    distr = np.random.lognormal

v = np.abs(distr(size = (n, m))) # buyers' valuations
B = np.abs(distr(size = (n,))) + 0.5 # buyers' budgets
# B = np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies

# some useful constants and functions
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on z[i] = v[i].dot(x[i])
L = np.max(B/u_min**2) # Lipschitz constant of sum of -B[i]*log(z[i])
Lf = L * np.linalg.norm(v, ord='fro') ** 2 # Lipschitz constant of f(x) = - sum B[i]*log(v[i].dot(x[i]))
# objective function
z = np.random.uniform(size=(n,))
neg_B_log_u_min = - B * np.log(u_min)
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

def h(z): # sum of -B[i] * log(z[i]) with extrap.
    is_above = (z >= u_min)
    above_array = -B * np.log(np.maximum(z, u_min))
    below_array = 0.5 * B/(u_min**2) * (z - u_min)**2 + (-B/u_min) * (z - u_min) + neg_B_log_u_min
    return np.sum(above_array * is_above + below_array * (1 - is_above))

def f(x): # actual obj. in optimization (with quad. extrapolation)
    return h(np.sum(v * x, 1))

def primal_obj(x): # original obj. assuming sum(x[:, j]) == s[j] for all j
    return -sum(B * np.log( np.sum(v * x, axis=1)))

def grad_f(x):
    """ compute z[i] = v[i].dot(x[i]) and gradient """
    z = np.sum(v * x, 1) # z = np.maximum(np.sum(v * x, 1), u_min)
    gg = np.zeros((n, m))
    for i in range(n):
        ghi = - B[i]/z[i] if z[i] >= u_min[i] else B[i]/(u_min[i]**2) * (z[i] - 2*u_min[i]) # derivative of quad. extrapolation
        gg[i, :] = ghi * v[i, :] # grad w.r.t. x[i], by chain rule
    return gg

def proj_and_multipliers(x): 
    """ project each column of x onto s[j]-simplex; return all multipliers too """
    results = [proj_simplex(x[:, j], s[j], return_multiplier=True) for j in range(m)]
    x_res, mult_res = np.array([qq[0] for qq in results]).T, np.array([qq[1] for qq in results])
    return x_res, mult_res

def dual_obj(p):
    """ the dual of EG primal (in maximization form) """
    # beta = np.min(1/v * p, axis = 1) # 
    beta = np.array([np.min(p/v[i, :]) for i in range(n)])
    return sum(B * np.log(beta)) - s.dot(p) - sum_B_log_B + sum_B

def eg_duality_gap(x, p):
    """ Given primal feasible x (>=0 and sum(x[:, j]) == s[j]) and dual feasible p (>=0), compute the EG duality cap """
    return primal_obj(x) - dual_obj(p)

##################################################################
print("======== solve the EG convex program using CVXPY + Mosek ========")
begin = time() # time it
x_cp = cp.Variable(shape=(n, m), nonneg=True)
obj_expr = 0 # sum of B[i] * log(v[i]' * x[i])
for i in range(n):
    obj_expr -= B[i] * cp.log(cp.matmul(v[i], x_cp[i]))
objective = cp.Minimize(obj_expr)
constraints = [cp.sum(x_cp[:, j]) <= s[j] for j in range(m)] # linear constraints
prob = cp.Problem(objective, constraints) # define the optimization problem
cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True) # solve it using SCS
p_cp = np.array([constraints[j].dual_value for j in range(m)])
time_elapased = time() - begin
print("min obj = {}, time elpased = {}".format(cvxpy_opt_obj, time_elapased))

def inner_prod_vertex(gg, vertex):
    """ gg is n-by-m, vv is a m-tuple of indices in {0, 1, ..., n-1} """
    return np.sum(gg[vertex, range(m)]) # sum og gg[vv[0], 0] + gg[vv[1], 1] + ... + gg[vv[m-1], m-1]

def vertex_to_mat(vertex): # convert vertex to a matrix
    xx = np.zeros((n, m))
    xx[vertex, range(m)] = 1
    return xx

def compute_x(repre):
    xx = np.zeros((n, m))
    for vertex, coeff in repre.items():
        xx[vertex, range(m)] += coeff
    return xx

##################################################################
print("======== Away-step Frank-Wolfe (linesearch: {}) ========".format(do_ls))
# construct initial x and its representation
vertex_init = tuple(np.random.randint(0, n, size=(m, )))
x = np.zeros(shape=(n,m))
x[vertex_init, range(m)] = 1
repre = defaultdict(lambda: 0)
repre[vertex_init] = 1
# num_init_vert = 5
# initial_vertices = [(ii,)*m for ii in range(num_init_vert)]
# initial_coeff = np.random.uniform(size = (num_init_vert, ))
# initial_coeff = initial_coeff / np.sum(initial_coeff)
# repre = defaultdict(lambda: 0, zip(initial_vertices, initial_coeff)) # any missing key (vertex) has value (coeff.) 0, by def. of representation
# x = np.zeros((n, m))
# for vv, cc in repre.items():
#     x[vv, range(m)] += cc

count_fw = 0
count_exact_ls = 0
for iter_idx in range(1, max_iter+1):
    # compute gradient
    gg = grad_f(x)
    vertex_fw = tuple(np.argmin(gg, axis=0)) # x[i,j] = 1 if i = x_fw_indices[j] and = 0 o.w.
    max_val = -np.inf
    for vertex in repre.keys():
        val = inner_prod_vertex(gg, vertex)
        if val > max_val:
            max_val, vertex_as = val, vertex # get max val. of <gg, vv>
    # choose FW or AS
    ggTx = np.sum(gg*x)
    use_FW = inner_prod_vertex(gg, vertex_fw) - ggTx <= ggTx - inner_prod_vertex(gg, vertex_as)
    if use_FW: # use FW
        count_fw += 1
        dd = vertex_to_mat(vertex_fw) - x
        gamma_max = 1
    else: # use AS
        dd = x - vertex_to_mat(vertex_as)
        # print(np.sum(gg*dd))
        coeff = repre[vertex_as]
        # if coeff <= 1e-3:
        #     break
        # print(coeff)
        gamma_max = coeff / (1 - coeff)
        # print("gamma_max = {}".format(gamma_max))
    ######### compute gamma #########
    if do_ls: # the search direction dd is either from FW or AS
        vd = np.sum(v*dd, axis = 1)
        vx = np.sum(v*x, axis = 1)
        one_dim_func = (lambda ll: - np.sum(B * vd/(vx + ll * vd)))
        # break
        # corner cases:
        if one_dim_func(0) > 0: # should not happen
            print("warning! Exact linesearch gives 0 step")
            gamma = 0
        elif one_dim_func(gamma_max) < 0:
            # print("use gamma_max")
            gamma = gamma_max
        else: # minimizer is in between
            count_exact_ls += 1
            gamma = bisect(one_dim_func, 0, gamma_max, disp=False)
    else: # just pick adaptive stepsize (Eq. (3.5) in Beck & Shtern)
        gamma = np.min([-np.sum(gg*dd)/(Lf * sum_squares(dd)), gamma_max])
    # update curr. iterate x
    x += gamma * dd
    # update representation of curr. iterate
    if use_FW: # use FW 
        for vv in repre.keys():
            repre[vv] *= (1 - gamma)
        repre[vertex_fw] += gamma
    else: # use AS
        # assert (vv in repre)
        for vv in repre.keys():
            repre[vv] *= (1 + gamma)
        repre[vertex_as] -= gamma
        if repre[vertex_as] <= 1e-5: # when gamma == gamma_max == mu/(1-mu), this is exactly 0
            del repre[vertex_as]
    if iter_idx % (max_iter//10) == 0:
        # compute p_diff
        tt = v * x
        b_ascg = ((tt.T / np.sum(tt, 1)) * B).T
        p_ascg = np.sum(b_ascg, axis=0)
        p_diff = np.max(np.abs(p_ascg - p_cp) / p_cp)
        print("iter = {}, FW steps = {}, exact ls = {}, obj = {:.5f}, p_diff = {:.5f}".format(iter_idx, count_fw, count_exact_ls, f(x), p_diff))

##################################################################
print("======== Original Frank-Wolfe (linesearch: {}) ========".format(do_ls))
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
count_exact_ls = 0
for iter_idx in range(1, max_iter+1):
    # compute gradient w.r.t. x
    gg = grad_f(x)
    # find the vertex that minimizes the gradient
    vertex_fw = tuple(np.argmin(gg, axis=0)) # x[i,j] = 1 if i = x_fw_indices[j] and = 0 o.w.
    # find stepsize
    if do_ls:
        vx = np.sum(v*x, axis=1)
        dd = vertex_to_mat(vertex_fw) - x
        vd = np.sum(v*dd, axis=1)
        one_dim_func = (lambda ll: - np.sum(B * vd/(vx + ll * vd)))
        if one_dim_func(0) > 0: # should not happen
            print("Warning: Exact linesearch gives gamma = 0")
            gamma = 0
        elif one_dim_func(1) < 0:
            # print("use gamma_max")
            gamma = 1
        else: # minimizer is in between
            count_exact_ls += 1
            gamma = bisect(one_dim_func, 0, 1, disp=False)
    else:
        gamma = 2 / (iter_idx + 2) # default stepsize
    # update x <-- x + gamma * (FW - x) == (1-gamma) * x + gamma * FW
    x = (1-gamma) * x
    x[vertex_fw, range(m)] += gamma
    # compute p_diff
    tt = v * x
    b_ascg = ((tt.T / np.sum(tt, 1)) * B).T
    p_ascg = np.sum(b_ascg, axis=0)
    p_diff = np.max(np.abs(p_ascg - p_cp) / p_cp)
    if p_diff <= accu:
        print("iter = {}, exact ls. = {}, obj = {:.5f}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, f(x), p_diff))
        break
    if iter_idx % (max_iter//10) == 0:
        print("iter = {}, exact ls. = {}, obj = {:.5f}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, f(x), p_diff))

print("======== prox. grad. with backtracking linesearch ========")
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on z[i] = v[i].dot(x[i])
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
p_diff = []
dgap_array = []
step = 1000/Lf # initial large stepsize
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx in range(1, max_iter + 1):
    gg = grad_f(x) # gradient
    obj_curr = f(x)
    for ls_idx in range(1, max_ls + 1):
        x_try, mult_try = proj_and_multipliers(x - step * gg)
        if step <= 1/Lf: # break if already very small step
            break # since safe step is very small, there can be numerical issues that lead to furtuer backtracking and make it extremely small
        obj_try = f(x_try)
        f_hat_try = obj_curr + np.sum(gg * (x_try - x)) + 1/(2*step) * sum_squares(x_try - x) # pseudo quad. approx. 
        if obj_try <= f_hat_try: # brek if sufficient decrease
            break
        # otherwise decrease step
        step *= bt_fac
    x, p = x_try, mult_try / step # update x and p
    total_bt += ls_idx # update total number of backtracking (minimum is 1)
    # print("iter = {}, num. of ls = {}, step = {}".format(iter_idx, ls_idx, step))
    total_bt_list.append(total_bt)
    if ls_idx == 1: # increase step if no backtracking is performed
        step *= inc_fac
        # print("increase step to {}".format(step))
    rel_diff_p = np.max(np.abs(p-p_cp)/p_cp) # relative difference in price
    dgap = eg_duality_gap(x, p) # EG duality gap
    # print(dgap, rel_diff_p)
    if iter_idx % max(max_iter//10, 1) == 0 or dgap <= 1e-5: # print occasionally
        obj_val = f(x)
        # rel_diff_x = np.linalg.norm(x - x_cp.value)/np.linalg.norm(x_cp.value) # diff. from the cvxpy optimal solution
        print("iter = {}, obj = {:.5f}, total bt. ls. = {}, p_diff = {:.5f}".format(iter_idx, obj_val, total_bt, dgap, rel_diff_p))
    # compute p_diff[t] and append to list
    p_diff.append(rel_diff_p)
    dgap_array.append(dgap)
    # if dgap <= accu: # no need to go further; otherwise may lead to numerical issues
    if rel_diff_p <= accu:
        print("iter = {}, obj = {:.5f}, total bt. ls. = {}, p_diff = {:.5f}".format(iter_idx, obj_val, total_bt, dgap, rel_diff_p))
        print("accu. reached at iter = {}, ls = {}".format(iter_idx, total_bt))
        break