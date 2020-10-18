''' PGLS for EG-Linear, PGLS, PR and FW for Shmyrev-Linear, assuming v[i,j] > 0 for all i,j '''

from utils import * 
import cvxpy as cp
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
    parser.add_argument('--varying_B', '--vb', type=int, default=0, help='varying B or all B = 1')
    args = parser.parse_args()
    # retrieve arguments
    n, m = args.n, args.m
    distr_name = args.distr_name
    file_mode = args.file_mode
    max_iter = args.max_iter
    sd = args.seed
    accu = args.desired_accuracy
    varying_B = bool(args.varying_B)
except:
    print("Error parsing arguments. Use default ones.")
    varying_B = False
    n, m, distr_name, file_mode, max_iter, sd, accu = 50, 100, 'unif', 'w', 5000, 1, 5e-6

print("varying B: {}".format(varying_B))
do_ls_fw = True # for FW algorithm
np.random.seed(sd)

###################################################################
# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join("logs/linear_varying_B", fname) if varying_B else os.path.join("logs/linear_B_1", fname)
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
# B = np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets # B = np.abs(distr(size = (n,))) + 0.5 # buyers' budgets
B = np.abs(distr(size = (n,))) + 0.5 if varying_B else np.ones(shape = (n,)) 
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies

# v[range(min(m,n)), range(min(m,n))] = 0

##################################################################
# some useful constants and functions
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on u[i] = v[i].dot(x[i])
L = np.max(B/u_min**2) # Lipschitz constant of sum of -B[i]*log(z[i])
# Lf_eg = L * np.linalg.norm(v, ord='fro') ** 2 # Lipschitz constant of EG f(x) = - sum B[i]*log(v[i].dot(x[i]))
Lf_eg = L * np.max(np.linalg.norm(v, axis=1))**2
# objective function
# z = np.random.uniform(size=(n,))
neg_B_log_u_min = - B * np.log(u_min)
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

p_upper_linear = sum_B # any optimal p[j] must <= total budget
denom_vec = np.sum(v, axis = 1)
# p_lower = np.array([np.max(v[:, j] * B / denom_vec) for j in range(m)])
temp_mat = (v.T * B / denom_vec).T # temp_mat[2, 3] == v[2,3]*B[2]/denom_vec[2]
p_lower_linear = np.max(temp_mat, axis = 0)
# Lipschitz constant for gradient, Lf = Lh * sig_max(A.T@A)
Lf_linear_shmyrev = 1/np.min(p_lower_linear) * (n * m)

def h_eg(z): # sum of -B[i] * log(z[i]) with extrap.
    is_above = (z >= u_min)
    above_array = -B * np.log(np.maximum(z, u_min))
    below_array = 0.5 * B/(u_min**2) * (z - u_min)**2 + (-B/u_min) * (z - u_min) + neg_B_log_u_min
    return np.sum(above_array[is_above]) + np.sum(below_array[np.logical_not(is_above)])

def f_eg(x): # actual EG obj. in optimization (with quad. extrapolation)
    return h_eg(np.sum(v * x, 1))

def primal_obj_eg(x): # original obj. assuming sum(x[:, j]) == s[j] for all j
    return -np.sum(B * np.log(np.sum(v * x, axis=1)))

def grad_f_eg(x):
    """ compute z[i] = v[i].dot(x[i]) and gradient """
    z = np.sum(v * x, 1)
    z[z==0] = -1 # to avoid NAN
    gh = - B/z*(z>=u_min) + B/(u_min**2)*(z-2*u_min)*(z<u_min)
    return (v.T*gh).T

def f_linear_shmyrev(b): # assuming b[i,j] == 0 if v[i,j] == 0
    pb = np.sum(b, axis = 0)
    return -np.sum(np.log(v)*b) + np.sum(pb * np.log(pb))

def grad_f_linear_shmyrev(b): # gradient of f(b) at any b > 0
    """ compute grad[i,j] = p[j]/b[i,j], where p[j] = sum of b[i,j] over i """
    gg = np.log(np.sum(b, axis = 0)/v) + 1
    return gg

def dual_obj(p):
    """ the dual of EG primal (in maximization form) """
    beta = np.min(p/v, axis = 1) # beta = np.array([np.min(p/v[i, :]) for i in range(n)])
    return np.sum(B * np.log(beta)) # assuming s[j] == 1

def eg_duality_gap(x, p):
    """ given primal feasible x (>=0 and sum(x[:, j]) == s[j]) and dual feasible p (>=0), compute the EG duality cap """
    return primal_obj_eg(x) - dual_obj(p)

def linear_shmyrev_dgap(b):
    ''' duality gap of any bid b (Shmyrev feasible) '''
    p = np.sum(b, axis=0)
    return - dual_obj(p) - np.sum(b*np.log(v)) + np.sum(p * np.log(p))

def compute_ave_dgap(b):
    """ an upper bound on ||p(t)-p_opt||_1 / m """
    return linear_shmyrev_dgap(b)/n
    # try NO square root / n

def compute_b_from_x(x):
    tt = v*x
    return ((tt.T / np.sum(tt, 1)) * B).T

# def proj_and_multipliers(x): # slow
#     """ project each column of x onto s[j]-simplex; return all multipliers too """
#     results = [proj_simplex(x[:, j], s[j], return_multiplier=True) for j in range(m)]
#     x_res, mult_res = np.array([qq[0] for qq in results]).T, np.array([qq[1] for qq in results])
#     return x_res, mult_res

print("======== proximal gradient on the EG primal (linear) with backtracking linesearch ========")
begin = time()
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on z[i] = v[i].dot(x[i])
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
dgap_array = []
ave_dgap_array = []
step = 100/Lf_eg # initial large stepsize
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx in range(1, max_iter + 1):
    gg = grad_f_eg(x) # gradient
    obj_curr = f_eg(x)
    for ls_idx in range(1, max_ls + 1):
        x_try, mult_try = proj_simplex_all(x - step * gg, s, return_multipliers=True) # 
        # x_try, mult_try = proj_and_multipliers(x - step * gg)
        if step <= 1/Lf_eg: # break if step is already small
            break # since safe step is very small, there can be numerical issues that lead to furtuer backtracking and make it extremely small
        obj_try = f_eg(x_try)
        f_hat_try = obj_curr + np.sum(gg * (x_try - x)) + 1/(2*step) * sum_squares(x_try - x) # pseudo quad. approx. 
        if obj_try <= f_hat_try: # brek if sufficient decrease
            break
        step *= bt_fac # otherwise decrease step
    x, p = x_try, mult_try / step # update x and p
    total_bt += ls_idx # update total number of backtracking (minimum is 1)
    total_bt_list.append(total_bt)
    if ls_idx == 1: # increase step if no backtracking is performed
        step *= inc_fac # print("increase step to {}".format(step))
    # compute residual measures and print
    b_eg = compute_b_from_x(x)
    # dgap = linear_shmyrev_dgap(b_eg)
    p_diff_ub = compute_ave_dgap(b_eg)
    if iter_idx % max(max_iter//10, 1) == 0:
        print("PGLS iter = {}, total ls. = {}, p_diff_ub = {:.5f}".format(iter_idx, total_bt, p_diff_ub))
    # compute p_diff[t] and append to list
    # dgap_array.append(dgap)
    ave_dgap_array.append(p_diff_ub)
    # if dgap <= accu: # no need to go further; otherwise may lead to numerical issues
    if p_diff_ub <= accu:
        print("PGLS p_diff_ub = {:.5f} <= {} at iter = {}, ls = {}".format(p_diff_ub, accu, iter_idx, total_bt))
        break

print("time elapsed = {}".format(time() - begin))
# save to file; both iter & ls counts are saved 
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls-linear-eg'] * max_iter, range(1, max_iter + 1), ave_dgap_array, total_bt_list)
csv_writer.writerows(rows)

##################################################################
print("======== Proportional Response (Wu & Zhang 2007, Zhang 2009, Birnbaum et al. 2011) ========")
begin = time()
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # initial (uniform) allocation, t = 0
# bids b[i,j] = x[i,j] * v[i,j]
b = v * x
dgap_array, ave_dgap_array = [], []
for iter_idx in range(1, max_iter+1):
    ###############  ###############
    p = np.sum(b, axis=0) # compute prices
    x = b / p # new allocation
    b = compute_b_from_x(x) # new bids
    # compute residuals and print
    p_diff_ub = compute_ave_dgap(b)
    if (iter_idx % (max_iter//10)) == 0:
        print("PR iter = {}, p_diff_ub = {:.5f}".format(iter_idx, p_diff_ub))
    # dgap_array.append(dgap), 
    ave_dgap_array.append(p_diff_ub)
    if p_diff_ub <= accu: # if dgap <= accu: # no need to go further...
        print("PR p_diff_ub <= {} reached at iter = {}".format(accu, iter_idx))
        break
time_elapased = time() - begin
print("time elapsed = {}".format(time_elapased))

# write to files
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr-linear-shmyrev'] * max_iter, range(1, max_iter+1), ave_dgap_array)
csv_writer.writerows(rows)

##################################################################
print("======== original Frank-Wolfe on EG (linesearch: {}) ========".format(do_ls_fw))
x = np.zeros(shape = (n, m)) # random initialization
x[range(n), range(n)] = 1
# x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
ave_dgap_array = []
count_exact_ls = 0
for iter_idx in range(1, max_iter+1):
    # compute gradient w.r.t. x and find the vertex that minimizes the gradient
    vertex_fw = tuple(np.argmin(grad_f_eg(x), axis=0)) # x[i,j] = 1 if i = x_fw_indices[j] and = 0 o.w.
    # find stepsize
    if do_ls_fw:
        vx = np.sum(v*x, axis=1)
        # dd = vertex_to_mat(vertex_fw) - x # the following two lines do the same
        dd = -x
        dd[vertex_fw, range(m)] += 1
        vd = np.sum(v*dd, axis=1)
        one_dim_func = (lambda ll: - np.sum(B * vd/(vx + ll * vd)))
        if one_dim_func(0) > 0: # should not happen
            print("Warning: FW exact linesearch gives gamma = 0")
            gamma = 0
        elif one_dim_func(1) < 0:
            print("Warning: FW use gamma_max")
            gamma = 1
        else: # minimizer is in between
            count_exact_ls += 1
            gamma = bisect(one_dim_func, 0, 1, disp=False)
    else:
        gamma = 2 / (iter_idx + 2) # default stepsize
    # update x <-- x + gamma * (FW - x) == (1-gamma) * x + gamma * FW
    x = (1-gamma) * x
    x[vertex_fw, range(m)] += gamma
    # compute residual measure
    b_fw = compute_b_from_x(x)
    ub_fw = compute_ave_dgap(b_fw)
    ave_dgap_array.append(ub_fw)
    if iter_idx % (max_iter//10) == 0:
        print("FWLS iter = {}, exact ls. = {}, p_diff_ub = {:.5f}".format(iter_idx, count_exact_ls, ub_fw))
    if ub_fw <= accu:
        print("FWLS ub_fw <= {} at iter {}, break".format(accu, iter_idx))
        break

# save to file
algo_name_fw = 'fwls-linear-eg' if do_ls_fw else 'fw-linear-eg'
rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, [algo_name_fw] * max_iter, range(1, max_iter+1), ave_dgap_array)
csv_writer.writerows(rows)

if False: # PGLS for linear Shmyrev does not seem to work very well
    ######################################################################################################
    print("======== solve (linear) Shmyrev using PGLS ========")
    ave_dgap_array = []
    b_pgls_linsh = np.outer(np.ones(m,), B/m).T
    step = 1000/Lf_linear_shmyrev # start from a large stepsize
    # dgap_array = []
    max_ls = 20 # max number of backtracking steps in linesearch
    bt_fac = 0.8 # backtracking discount factor
    inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
    total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
    total_bt_list = []
    for iter_idx_pgls in range(1, max_iter+1):
        obj_curr, gg = f_linear_shmyrev(b_pgls_linsh), grad_f_linear_shmyrev(b_pgls_linsh)
        for ls_idx in range(1, max_ls + 1): # linesearch iterations
            b_try = proj_simplex_all((b_pgls_linsh - step * gg).T, B).T
            if step <= 1/Lf_linear_shmyrev:
                break
            obj_try = f_linear_shmyrev(b_try)
            f_hat_try = obj_curr + np.sum(gg * (b_try - b_pgls_linsh)) + 1/(2*step) * sum_squares(b_try - b_pgls_linsh)
            if obj_try <= f_hat_try:
                break
            step *= bt_fac # print("iter = {}, step = {}".format(iter_idx_pgls, step))
        b_pgls_linsh = b_try
        total_bt += ls_idx
        total_bt_list.append(total_bt)
        if ls_idx == 1: # increase step if no backtracking is performed
            step *= inc_fac
        ub_pgls_linsh = compute_ave_dgap(b_pgls_linsh)
        ave_dgap_array.append(ub_pgls_linsh)
        if ub_pgls_linsh <= accu:
            print("at iter = {}, total_ls = {}, ub_pgls_ql = {:.5f} <= {}".format(iter_idx_pgls, total_bt, ub_pgls_linsh, accu))
            break
        if iter_idx_pgls % (max_iter//5) == 0:
            print("iter = {}, total_ls = {}, ub_pgls_ql = {:.5f}".format(iter_idx_pgls, total_bt, ub_pgls_linsh))

    # write running logs to file
    rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls-linear-shmyrev'] * max_iter, range(1, max_iter + 1), ave_dgap_array, total_bt_list)
    csv_writer.writerows(rows)

ff.close() # close file