#############################################################################
#############################################################################
#############################################################################
""" load module, define constants and functions"""

from utils import *
# import cvxpy as cp
from time import time
from scipy.optimize import bisect
import os, csv, argparse

#######################################################################################
# arguments for experiment setups
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
    parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
    parser.add_argument('--m', type=int, default=100, help='m = number of items')
    parser.add_argument('--max_iter', type=int, default=20000, help='max number of iterations')
    parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
    parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
    parser.add_argument('--desired_accuracy', '--da', type=float, default=5e-6, help='stop early when dgap or p_diff <= da')
    args = parser.parse_args()
    # retrieve arguments
    n, m = args.n, args.m
    distr_name = args.distr_name
    file_mode = args.file_mode
    max_iter = args.max_iter
    sd = args.seed
    accu = args.desired_accuracy
except:
    print("Error parsing arguments. Use default ones.")
    n, m, distr_name, file_mode, max_iter, sd, accu = 50, 100, 'unif', 'w', 20000, 1, 5e-6

do_ls_fw = True # for FW algorithm
np.random.seed(sd)

###################################################################
# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join("logs/ql", fname)
try:
    print("file mode (w or a): {}".format(file_mode))
    ff = open(fpath, file_mode)
except:
    ff = open(fpath, 'w')
csv_writer = csv.writer(ff)

print("n = {}, m = {}, seed = {}, distr. of v = {}, accu = {}".format(n, m, sd, distr_name, accu))

# distribution for random matrix generation
if distr_name == "unif":
    distr = np.random.uniform # np.random.exponential
elif distr_name == "exp":
    distr = np.random.exponential
elif distr_name == 'normal': 
    distr = np.random.normal
elif distr_name in {'log_normal', 'lognormal', 'ln'}:
    distr = np.random.lognormal

v = np.abs(distr(size = (n, m))) # buyers' valuations
# B = 5 * np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets # B = np.abs(distr(size = (n,))) + 0.5 # buyers' budgets
B = 5 * (1 + np.abs(distr(size = (n,))))
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies

##################################################################
# some useful constants and functions
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on u[i] = v[i].dot(x[i])
L = np.max(B/u_min**2) # Lipschitz constant of sum of -B[i]*log(z[i])
Lf_eg = L * np.max(np.linalg.norm(v, axis=1)) ** 2 # Lipschitz constant of EG f(x) = - sum B[i]*log(v[i].dot(x[i]))
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
########### Lipschitz constant for gradient, Lf = Lh * sig_max(A.T@A)
Lf_ql_shmyrev = 1/np.min(p_lower_ql) * n

def f_ql_shmyrev(b):
    ''' the QL-Shmyrev minimization objective in bids b = (b[i,j])'''
    pb = np.sum(b, axis = 0)
    return -np.sum(((1 + np.log(v)) * b)) + np.sum(pb * np.log(pb))

def grad_f_ql_shmyrev(b): # gradient of f(b) at any b > 0
    """ compute grad[i,j] = p[j]/b[i,j], where p[j] = sum of b[i,j] over i """
    return np.log(np.sum(b, axis = 0)/v)

neg_entropy_p_lower = p_lower_ql * np.log(p_lower_ql) 
one_plus_log_p_lower = 1+np.log(p_lower_ql)
one_over_p_lower = 1/p_lower_ql
def f_ql_shmyrev_extrapolated(b):
    pb = np.sum(b, axis = 0)
    term1 = -np.sum((1 + np.log(v)) * b)
    term21, term22 = pb * np.log(pb), neg_entropy_p_lower + one_plus_log_p_lower * (pb - p_lower_ql) + 0.5 * one_over_p_lower * (pb - p_lower_ql)**2
    term2 = np.sum(term21[pb>=p_lower_ql]) + np.sum(term22[pb<p_lower_ql])
    return term1 + term2

def grad_f_ql_shmyrev_extrapolated(b):
    term1 = - (1 + np.log(v))
    pb = np.sum(b, axis = 0)
    tt1 = 1 + np.log(pb)
    tt2 = one_plus_log_p_lower + one_over_p_lower * (pb - p_lower_ql)
    grad_h = np.zeros((m,))
    geq_mask = (pb >= p_lower_ql)
    grad_h[geq_mask], grad_h[np.logical_not(geq_mask)] = tt1[geq_mask], tt2[np.logical_not(geq_mask)]
    # grad_h = tt1 * (pb >= p_lower_ql) + tt2 * (pb < p_lower_ql)
    return term1 + np.outer(np.ones((n,)), grad_h)

def dual_obj_ql(p): # QL dual in (p, beta)
    beta = np.minimum(np.min(p/v, axis=1), 1)
    return np.sum(p) - np.sum(B * np.log(beta))

def ql_shmyrev_dgap(b):
    # obj. of the dual in terms of (p, beta)
    pb = np.sum(b, axis = 0)
    return f_ql_shmyrev(b) + dual_obj_ql(pb)

def ave_p_diff_ub(b):
    """ an upper bound on ||p(t)-p_opt||_1 / m """
    # return np.sqrt(2*ql_shmyrev_dgap(b))/n
    return ql_shmyrev_dgap(b)/n

######################################################################################################
print("======== solve QL-Shmyrev using PGLS ========")
ave_ub_array = []
b_aug = np.outer(np.ones(m+1,), B/(m+1)).T # b_aug := (b, delta)
g_aug = np.zeros((n, m+1)) # to store gradient corr. to b_aug, which is 0 for the delta part
step = 100/Lf_ql_shmyrev # start from a large stepsize
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx_pgls in range(1, max_iter+1):
    # print("iter = {}".format(iter_idx_pgls))
    g_aug[:, :m] = grad_f_ql_shmyrev_extrapolated(b_aug[:, :m]) # compute grad. vec. of the b part (grad. for the delta part is 0)
    obj_curr = f_ql_shmyrev_extrapolated(b_aug[:, :m]) # compute current objective
    for ls_idx in range(1, max_ls + 1): # linesearch iterations
        b_aug_try = proj_simplex_all((b_aug - step * g_aug).T, B).T # b_aug_try = np.stack([proj_simplex(b_aug[i, :] - step * g_aug[i, :], s=B[i]) for i in range(n)])
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
    # p_pgls_ql = np.sum(b_aug[:, :m], axis=0) # compute prices
    b_pgls_ql = b_aug[:, :m]
    p_pgls = np.sum(b_aug[:, :m], axis=0) # compute prices
    num_lower = np.sum(p_pgls <= p_lower_ql)
    if num_lower >= 1:
        print("iter = {}, num_lower = {}".format(iter_idx_pgls, num_lower))
    ub_pgls_ql = ave_p_diff_ub(b_pgls_ql)
    ave_ub_array.append(ub_pgls_ql)
    if iter_idx_pgls % (max_iter//10) == 0:
        print("PGLS iter = {}, total_ls = {}, ub_pgls_ql = {:.5f}".format(iter_idx_pgls, total_bt, ub_pgls_ql))
    if ub_pgls_ql <= accu:
        print("PGLS at iter = {}, total_ls = {}, ub_pgls_ql = {:.5f} <= {}".format(iter_idx_pgls, total_bt, ub_pgls_ql, accu))
        break

# write running logs to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls-ql-shmyrev'] * max_iter, range(1, max_iter + 1), ave_ub_array, total_bt_list)
csv_writer.writerows(rows)

######################################################################################################
print("======== solve QL-Shmyrev using PR dynamics ========")
# set initial iterates
begin = time() # time it
b_prql = np.outer(np.ones(m, ), B/(m+1)).T # initial bids
delta = B/(m+1) # initial leftover
# max_iter = 5000 # use input argument
ave_ub_array = []
for iter_idx_prql in range(1, max_iter+1):
    p_prql = np.sum(b_prql, axis=0) # prices
    x_pr = b_prql / p_prql # allocations
    # bids and delta
    tt = v * x_pr
    denom_vec = np.sum(tt, axis=1) + delta
    b_prql = ((tt.T / denom_vec) * B).T
    delta = B * delta / denom_vec
    # compute the ave_ub for price diff.
    ub_prql = ave_p_diff_ub(b_prql)
    ave_ub_array.append(ub_prql) # print(ql_shmyrev_dgap(b_prql))
    if iter_idx_prql % (max_iter//10) == 0:
        print("PR iter = {}, ub_prql = {}".format(iter_idx_prql, ub_prql))
    if ub_prql <= accu:
        print("PR p_diff_ub <= {} at iter = {}, break".format(accu, iter_idx_prql))
        break
time_elapased = time() - begin
print("prql time elpased = {}".format(time_elapased))

# write to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr-ql-shmyrev'] * max_iter, range(1, max_iter + 1), ave_ub_array)
csv_writer.writerows(rows)

# #############################################################################
# do_ls_fw = True
# print("======== original Frank-Wolfe on QL-Shmyrev (linesearch: {}) ========".format(do_ls_fw))
# # initial iterate
# b_aug = np.zeros((n, m+1))
# b_aug[range(n), range(n)] = 1
# b_aug[range(n), range(n, m)] = 1
# b_aug[range(n), m] = 1
# b_aug /= 3
# ave_ub_array = []
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
#         dd_reduced, b_reduced = dd[:, :m], b_aug[:, :m]
#         pd = np.sum(dd_reduced, axis = 0)
#         pb = np.sum(b_reduced, axis = 0)
#         first_term = -np.sum((1+np.log(v)) * dd_reduced) + np.sum(pd)
#         one_dim_func = lambda ll: first_term + np.sum(pd * np.log(pb + ll * pd))
#         if one_dim_func(0) > 0: # should not happen
#             print("Warning: Exact linesearch gives gamma = 0")
#             gamma = 0
#         elif one_dim_func(1) < 0:
#             print("Warning: use gamma_max = 1")
#             gamma = 1
#         else: # minimizer is in between
#             count_exact_ls += 1
#             gamma = bisect(one_dim_func, 0, 1, disp=False)
#     else:
#         gamma = 2 / (iter_idx + 2) # default stepsize
#     # update x <-- x + gamma * (FW - x) == (1-gamma) * x + gamma * FW
#     b_aug = (1-gamma) * b_aug
#     b_aug[range(n), vertex_fw] += gamma * B # non-uniform B
#     ub_fw = ave_p_diff_ub(b_aug[:, :m])
#     ave_ub_array.append(ub_fw)
#     # print("iter = {}, exact ls. = {}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, p_diff_fw))
#     if iter_idx % (max_iter//10) == 0:
#         print("FW iter = {}, exact ls. = {}, ub_fw = {:.5f}".format(iter_idx, count_exact_ls, ub_fw))
#     if ub_fw <= accu:
#         print("FW ub_fw <= {} at iter {}, break".format(accu, iter_idx))
#         break

# # # save to file
# algo_name_fw = 'fwls-ql-shmyrev' if do_ls_fw else 'fw-ql-shmyrev'
# rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, [algo_name_fw] * max_iter, range(1, max_iter+1), ave_ub_array)
# csv_writer.writerows(rows)

####################################################################
ff.close() # close file