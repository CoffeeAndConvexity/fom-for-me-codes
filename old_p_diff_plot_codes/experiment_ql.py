# trying out PGLS and PR (MD) for QL-Shmyrev

""" load module, define constants and functions"""
from utils import * 
import cvxpy as cp
from time import time
from scipy.optimize import bisect
import os, csv, argparse

try:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default='logs/ql/', help='log file mode (w or a)')
    parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
    parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
    parser.add_argument('--m', type=int, default=100, help='m = number of items')
    parser.add_argument('--max_iter', type=int, default=20000, help='max number of iterations')
    parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
    parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
    parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-3, help='stop early when dgap <= da')
    args = parser.parse_args()
    # retrieve arguments
    n, m, distr_name, file_mode = args.n, args.m, args.distr_name, args.file_mode
    max_iter, sd, accu = args.max_iter, args.seed, args.desired_accuracy
except:
    print("Warning: cannot parse arguments. use default ones")
    n, m, distr_name, file_mode, max_iter, sd, accu = 50, 100, 'unif', 'w', 20000, 1, 1e-3

np.random.seed(sd)

###################################################################
# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join('logs/ql', fname)
try:
    print("file mode (w or a): {}".format(file_mode))
    ff = open(fpath, file_mode)
except:
    ff = open(fpath, 'w')
csv_writer = csv.writer(ff)

print("n = {}, m = {}, seed = {}, distr. of v = {}, accu_tol = {}".format(n, m, sd, distr_name, accu))

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
# B = np.ones(shape=(n,)) # B[i] too small, all buyers spend all budget
B = 5 * (1 + np.abs(distr(size = (n,)))) # partially spent with leftover
# assume s[i] = 1 for all i

##################################################################
# some useful constants and functions
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on u[i] = v[i].dot(x[i])
L = np.max(B/u_min**2) # Lipschitz constant of sum of -B[i]*log(z[i])
# objective function
z = np.random.uniform(size=(n,))
neg_B_log_u_min = - B * np.log(u_min)
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

p_upper_ql = np.max(v, axis = 0) # this is only for QL; not for linear
denom_vec = np.sum(v, axis = 1) + B
# p_lower = np.array([np.max(v[:, j] * B / denom_vec) for j in range(m)])
temp_mat = (v.T * B / denom_vec).T # temp_mat[2, 3] == v[2,3]*B[2]/denom_vec[2]
p_lower_ql = np.max(temp_mat, axis = 0)
# Lipschitz constant for gradient, Lf = Lh * sig_max(A.T@A)
# Lf_ql_shmyrev = 1/np.min(p_lower_ql) * (n * m) # old way, incorrect (too loose)
Lf_ql_shmyrev = 1/np.min(p_lower_ql) * n

def f_ql_shmyrev(b):
    ''' the QL-Shmyrev minimization objective in bids b = (b[i,j])'''
    pb = np.sum(b, axis = 0)
    return - np.sum((1 + np.log(v)) * b) + np.sum(pb * np.log(pb))

def grad_f_ql_shmyrev(b): # gradient of f_ql_shmyrev(b) at any b > 0
    return np.log(np.sum(b, axis = 0)/v)

neg_entropy_p_lower, neg_entropy_p_upper = p_lower_ql * np.log(p_lower_ql), p_upper_ql * np.log(p_upper_ql)
one_plus_log_p_lower, one_plus_log_p_upper = 1+np.log(p_lower_ql), 1+np.log(p_upper_ql)
one_over_p_lower, one_over_p_upper = 1/p_lower_ql, 1/p_upper_ql
def f_ql_shmyrev_extrapolated(b):
    pb = np.sum(b, axis = 0)
    term1 = -np.sum((1 + np.log(v)) * b)
    term21 = pb * np.log(np.maximum(pb, p_lower_ql))
    term22 = neg_entropy_p_lower + one_plus_log_p_lower * (pb - p_lower_ql) + 0.5 * one_over_p_lower * (pb - p_lower_ql)**2
    term23 = neg_entropy_p_upper + one_plus_log_p_upper * (pb - p_upper_ql) + 0.5 * one_over_p_upper * (pb - p_upper_ql)**2
    term2 = np.sum(term21[(p_lower_ql<=pb)*(pb<=p_upper_ql)]) + np.sum(term22[pb<p_lower_ql]) + np.sum(term23[pb>p_upper_ql])  # in corner cases first term might have nan values, which are masked
    return term1 + term2
# def grad_f_ql_shmyrev_extrapolated(b):
#     term1 = - (1 + np.log(v))
#     pb = np.sum(b, axis = 0)
#     tt1 = 1 + np.log(pb)
#     tt2 = one_plus_log_p_lower + one_over_p_lower * (pb - p_lower_ql)
#     grad_h = np.zeros((m,))
#     geq_mask = (pb >= p_lower_ql)
#     grad_h[geq_mask], grad_h[np.logical_not(geq_mask)] = tt1[geq_mask], tt2[np.logical_not(geq_mask)]
#     # grad_h = tt1 * (pb >= p_lower_ql) + tt2 * (pb < p_lower_ql)
#     return term1 + np.outer(np.ones((n,)), grad_h)
def grad_f_ql_shmyrev_extrapolated(b):
    term1 = - (1 + np.log(v))
    pb = np.sum(b, axis = 0)
    tt1 = 1 + np.log(np.maximum(pb, p_lower_ql))
    tt2 = one_plus_log_p_lower + one_over_p_lower * (pb - p_lower_ql)
    tt3 = one_plus_log_p_upper + one_over_p_upper * (pb - p_upper_ql)
    grad_h = tt1 * ((pb >= p_lower_ql) * (pb <= p_upper_ql)) + tt2 * (pb < p_lower_ql) + tt3 * (pb > p_upper_ql)
    return term1 + np.outer(np.ones((n,)), grad_h)

def duality_gap(p, b): # given p and b, compute duality gap
    # obj. of the dual in terms of (p, beta)
    beta = np.min(p/v, 1)
    obj_dual = np.sum(p) - np.sum(B * np.log(beta))
    obj_primal = f_ql_shmyrev(b)
    return obj_dual + obj_primal # - np.sum(B * (1 - np.log(B)))

tt = -B * np.log(B) + B
def f_ql_eg(x):
    u = np.sum(v*x, 1)
    geq_B = (u>=B)
    return -np.sum((B * np.log(u))[geq_B]) + np.sum((tt-z)[np.logical_not(geq_B)])

def grad_f_ql_eg(x):
    u = np.sum(v*x, 1)
    grad_h = - np.maximum(B/u, 1)
    gg = np.zeros(shape = (n, m))
    for i in range(m):
        gg[:, i] = grad_h[i] * v[i]
    return gg

print("======== solve QL-Shmyrev using Mosek ========")
# Cole et al. (2016) Lemma 5
begin = time() # time it
b_cp = cp.Variable(shape = (n, m), nonneg=True) # b_cp
p_cp = cp.Variable(shape = (m, ), nonneg=True) # prices - can be eliminated
obj_expr = -cp.sum(cp.multiply(1 + np.log(v), b_cp)) - cp.sum(cp.entr(p_cp)) # keep the constant so that objective values match
objective = cp.Minimize(obj_expr)
constraints = [cp.sum(b_cp[:, j]) == p_cp[j] for j in range(m)] # p[j] := sum of b[i,j] over all i
constraints.extend([cp.sum(b_cp[i, :]) <= B[i] for i in range(n)]) # budget constraints
prob = cp.Problem(objective, constraints) # define the optimization problem
cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True)
price_via_shmyrev_mosek = p_cp.value
x_shmyrev = b_cp.value / p_cp.value
time_elapased = time() - begin
print("min obj = {}, time elpased = {}".format(cvxpy_opt_obj, time_elapased))

print("======== solve QL-Shmyrev using PR dynamics ========")
# set initial iterates
begin = time() # time it
b_pr = np.outer(np.ones(m, ), B/(m+1)).T # initial bids
delta = B/(m+1) # initial leftover
# max_iter = 5000 # use input argument
p_diff_array_pr = []
dgap_array_pr = []
for iter_idx_pr in range(1, max_iter+1):
    p_pr = np.sum(b_pr, axis=0) # prices
    x_pr = b_pr / p_pr # allocations
    # bids
    tt = v * x_pr
    denom_vec = np.sum(tt, axis=1) + delta
    b_pr = ((tt.T / denom_vec) * B).T
    delta = B * delta / denom_vec
    p_diff_pr = np.max(np.abs(p_pr - price_via_shmyrev_mosek)/price_via_shmyrev_mosek)
    p_diff_array_pr.append(p_diff_pr)
    if p_diff_pr <= accu:
        print("p_diff <= {} at iter = {}, break".format(accu, iter_idx_pr))
        break
    # print("iter = {}, obj val. = {}, p_diff = {}".format(iter_idx_pr, f_ql_shmyrev(b_pr), p_diff_pr))
    if iter_idx_pr % (max_iter//10) == 0:
        print("iter = {}, obj val. = {}, p_diff = {}".format(iter_idx_pr, f_ql_shmyrev(b_pr), p_diff_pr))
    # for i in range(n): # the old way
    #     denom = (np.sum(tt[i]) + delta[i])
    #     b[i] = B[i] * tt[i] / denom
    #     delta[i] = B[i] * delta[i] / denom
time_elapased = time() - begin
print("time elpased = {}".format(time_elapased))
rel_diff_prices = np.max(np.abs((p_pr - price_via_shmyrev_mosek)/price_via_shmyrev_mosek))

# write to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr'] * max_iter, range(1, max_iter + 1), p_diff_array_pr)
csv_writer.writerows(rows)

print("======== solve QL-Shmyrev using PGLS ========")
p_diff_array_pgls = []
b_aug = np.outer(np.ones(m+1, ), B/(m+1)).T # b_aug = (b, delta)
g_aug = np.zeros((n, m+1)) # to store gradient
step = 100/Lf_ql_shmyrev # start from a large stepsize
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx_pgls in range(1, max_iter+1):
    g_aug[:, :m] = grad_f_ql_shmyrev_extrapolated(b_aug[:, :m]) # compute grad. vec. of the b part (grad. for the delta part is 0)
    # g_aug[:, :m] = grad_f_ql_shmyrev(b_aug[:, :m])
    obj_curr = f_ql_shmyrev_extrapolated(b_aug[:, :m]) # compute current objective
    for ls_idx in range(1, max_ls + 1): # linesearch iterations
        # b_aug_try = np.stack([proj_simplex(b_aug[i, :] - step * g_aug[i, :], s=B[i]) for i in range(n)])
        b_aug_try = proj_simplex_all((b_aug - step * g_aug).T, B).T
        if step <= 1/Lf_ql_shmyrev:
            break
        obj_try = f_ql_shmyrev_extrapolated(b_aug_try[:, :m])
        f_hat_try = obj_curr + np.sum(g_aug * (b_aug_try - b_aug)) + 1/(2*step) * sum_squares(b_aug_try - b_aug)
        if obj_try <= f_hat_try:
            break
        # print("iter = {}, step = {}".format(iter_idx_pgls, step))
        step *= bt_fac
    b_aug = b_aug_try
    total_bt += ls_idx
    total_bt_list.append(total_bt)
    if ls_idx == 1: # increase step if no backtracking is performed
        step *= inc_fac
    p_pgls = np.sum(b_aug[:, :m], axis=0) # compute prices
    num_lower = np.sum(p_pgls <= p_lower_ql)
    if num_lower >= 1:
        print("iter = {}, num_lower = {}".format(iter_idx_pgls, num_lower))
    b_pgls = b_aug[:, :m]
    p_diff_pgls = np.max(np.abs(p_pgls-price_via_shmyrev_mosek)/price_via_shmyrev_mosek) # relative difference in price
    p_diff_array_pgls.append(p_diff_pgls)
    if p_diff_pgls <= accu:
        print("p_diff <= {} at iter = {}, ls = {}, break".format(accu, iter_idx_pgls, total_bt))
        break
    if iter_idx_pgls % (max_iter//10) == 0:
        print("iter = {}, total_ls = {}, obj val. = {}, rel_diff_p = {}".format(iter_idx_pgls, total_bt, f_ql_shmyrev(b_aug[:, :m]), p_diff_pgls))

# write running logs to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls'] * max_iter, range(1, max_iter + 1), p_diff_array_pgls, total_bt_list)
csv_writer.writerows(rows)

# #############################################################################
# do_ls_fw = True
# begin = time()
# print("======== original Frank-Wolfe on QL-Shmyrev (linesearch: {}) ========".format(do_ls_fw))
# # initial iterate
# b_aug = np.zeros((n, m+1))
# b_aug[range(n), range(n)] = 1
# b_aug[range(n), range(n, m)] = 1
# b_aug[range(n), m] = 1
# b_aug /= 3
# p_diff_array = []
# g_aug = np.zeros((n, m+1)) # to store gradient
# count_exact_ls = 0
# for iter_idx in range(1, max_iter+1):
#     # compute gradient w.r.t. b
#     g_aug[:, :m] = grad_f_ql_shmyrev_extrapolated(b_aug[:, :m])
#     # find the vertex that minimizes the gradient
#     vertex_fw = tuple(np.argmin(g_aug, axis=1)) # x[i,j] = 1 if i = x_fw_indices[j] and = 0 o.w.
#     # find stepsize
#     if do_ls_fw: # use original function here, wlog
#         # dd = vertex_to_mat(vertex_fw) - x # the following two lines do the same
#         dd = -b_aug
#         dd[range(n), vertex_fw] += B
#         # print("total num. of \'active\' delta[i]: {}".format(np.sum(np.array(vertex_fw)==m)))
#         dd_reduced, b_reduced = dd[:, :m], b_aug[:, :m]
#         pd = np.sum(dd_reduced, axis = 0)
#         pb = np.sum(b_reduced, axis = 0)
#         first_term = -np.sum((1+np.log(v)) * dd_reduced) + np.sum(pd)
#         one_dim_func = lambda ll: first_term + np.sum(pd * np.log(pb + ll * pd))
#         # one_dim_func = lambda ll: np.sum(grad_f_ql_shmyrev_extrapolated(b_reduced + ll * dd_reduced)*dd_reduced)
#         if one_dim_func(0) > 0: # should not happen
#             print("Warning: Exact linesearch gives gamma = 0")
#             gamma = 0
#         elif one_dim_func(1) < 0:
#             print("Warning: use gamma_max = 1")
#             gamma = 1
#         else: # minimizer is in between
#             count_exact_ls += 1
#             gamma = bisect(one_dim_func, 0, 1, disp=False)
#         if gamma == 0 or gamma == 1:
#             gamma = 2 / (iter_idx + 2)
#     else:
#         gamma = 2 / (iter_idx + 2) # default stepsize
#     # update x <-- x + gamma * (FW - x) == (1-gamma) * x + gamma * FW
#     b_aug = (1-gamma) * b_aug
#     b_aug[range(n), vertex_fw] += gamma * B
#     p_fw = np.sum(b_aug[:, :m], 0)
#     p_diff_fw = np.max(np.abs(p_fw - price_via_shmyrev_mosek)/price_via_shmyrev_mosek)
#     p_diff_array.append(p_diff_fw)
#     # print("iter = {}, exact ls. = {}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, p_diff_fw))
#     if iter_idx % (max_iter//10) == 0:
#         print("iter = {}, exact ls. = {}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, p_diff_fw))
#     if p_diff_fw <= accu:
#         print("p_diff_fw <= {} at iter {}, break".format(accu, iter_idx))
#         break

# print("fw time = {}".format(time()-begin))
# # # save to file
# algo_name_fw = 'fwls' if do_ls_fw else 'fw'
# rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, [algo_name_fw] * max_iter, range(1, max_iter+1), p_diff_array)
# csv_writer.writerows(rows)

# close file
ff.close()