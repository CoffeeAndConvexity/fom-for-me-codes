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
    parser.add_argument('--distr_name', '--dn', type=str, default='normal', help='data generation distibution name (unif, normal, exp, log_normal)')
    parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
    parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-4, help='stop early when dgap or p_diff <= da')
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
    varying_B = True
    n, m, distr_name, file_mode, max_iter, sd, accu = 500, 1000, 'normal', 'w', 20000, 1, 1e-4

print("varying B: {}".format(varying_B))
np.random.seed(sd)

###################################################################
# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join("logs/linear_time_it/", fname)
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
# assume s[j] == 1 

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

def primal_obj_eg(x): # original obj. assuming sum(x[:, j]) == s[j] for all j
    return -np.sum(B * np.log(np.sum(v * x, axis=1)))

# @jit(nopython=True)
def f_eg(x): # actual obj. func. in optimization (quad. extrapolated)
    return h_eg(np.sum(v * x, 1))

# @jit(nopython=True)
def grad_f_eg(x):
    """ compute z[i] = v[i].dot(x[i]) and gradient """
    uu = np.sum(v * x, 1)
    uu[uu==0] = -1 # to avoid NAN
    gh = - B/uu*(uu>=u_min) + B/(u_min**2)*(uu-2*u_min)*(uu<u_min)
    return (v.T*gh).T

# @jit(nopython=True)
def f_and_grad_eg(x):
    uu = np.sum(v * x, 1)
    uu[uu==0] = -1 # to avoid NAN
    is_above, is_below = (uu >= u_min), (uu<u_min)
    term1 = -B[is_above] * np.log(uu[is_above])
    term2 = 0.5 * B[is_below]/(u_min[is_below]**2) * (uu[is_below] - u_min[is_below])**2 + (-B[is_below]/u_min[is_below]) * (uu[is_below] - u_min[is_below]) + neg_B_log_u_min[is_below]
    gh = - B/uu*(uu>=u_min) + B/(u_min**2)*(uu-2*u_min)*(uu<u_min)
    return np.sum(term1) + np.sum(term2), (v.T*gh).T

def f_linear_shmyrev(b): # assuming b[i,j] == 0 if v[i,j] == 0
    pb = np.sum(b, axis = 0)
    return -np.sum(np.log(v)*b) + np.sum(pb * np.log(pb))

def grad_f_linear_shmyrev(b): # gradient of f(b) at any b > 0
    """ compute grad[i,j] = p[j]/b[i,j], where p[j] = sum of b[i,j] over i """
    gg = np.log(np.sum(b, axis = 0)/v) + 1
    return gg

# @jit(nopython=True)
def dual_obj(p):
    """ the dual of EG primal (in maximization form) """
    beta = np.min(p/v, axis = 1) # beta = np.array([np.min(p/v[i, :]) for i in range(n)])
    return np.sum(B * np.log(beta)) # assuming s[j] == 1

# @jit(nopython=True)
def eg_duality_gap(x, p):
    """ given primal feasible x (>=0 and sum(x[:, j]) == s[j]) and dual feasible p (>=0), compute the EG duality cap """
    return primal_obj_eg(x) - dual_obj(p)

def linear_shmyrev_dgap(b):
    ''' duality gap of any bid b (Shmyrev feasible) '''
    p = np.sum(b, axis=0)
    return - dual_obj(p) - np.sum(b*np.log(v)) + np.sum(p * np.log(p)) #+ sum_B_log_B

def compute_ave_dgap(b):
    """ an upper bound on ||p(t)-p_opt||_1 / m """
    return linear_shmyrev_dgap(b)/n
    # try NO square root / n

def compute_b_from_x(x):
    tt = v*x
    return ((tt.T / np.sum(tt, 1)) * B).T

# ##################################################################
# if False:
#     print("======== solve the EG convex program using CVXPY + Mosek ========")
#     begin = time() # time it
#     x_cp = cp.Variable(shape=(n, m), nonneg=True)
#     obj_expr = 0 # sum of B[i] * log(v[i]' * x[i])
#     for i in range(n):
#         obj_expr -= B[i] * cp.log(cp.matmul(v[i], x_cp[i]))
#     objective = cp.Minimize(obj_expr)
#     constraints = [cp.sum(x_cp[:, j]) == 1 for j in range(m)] # linear constraints
#     prob = cp.Problem(objective, constraints) # define the optimization problem
#     cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True, verbose=False) #, mosek_params= {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-4}) # solve it
#     x_cp = x_cp.value
#     p_cp = np.array([constraints[j].dual_value for j in range(m)])
#     time_cp = time() - begin
#     print("min obj = {}, time time_cp = {}".format(cvxpy_opt_obj, time_cp))
#     b_cp = compute_b_from_x(x_cp) # new bids
#     dgap_cp = compute_ave_dgap(b_cp)
#     print("ave_dgap_cp = {}".format(dgap_cp))

v = v.astype(np.float32)
B = B.astype(np.float32)

################################################################################################
##################################### call Mosek directly ###################################
import scipy as sp
from scipy import sparse
import scipy.linalg as spla

v_part = sparse.block_diag([v[i] for i in range(n)])
I_rep = sparse.hstack([sparse.identity(m) for i in range(n)])
In = sparse.identity(n)
A_full = sparse.hstack([v_part, -In])
temp = sparse.hstack([I_rep, sparse.csc_matrix(([], ([],[])), shape=(m, n))])
A_full = sparse.vstack([A_full, temp])
A_full = sparse.block_diag([A_full, In])
temp = sparse.csc_matrix(([], ([],[])), shape=(2*n+m, n))
A_full = sparse.hstack([A_full, temp]) # A_full.shape == (2*n+m, n*m+3*n )
# A_full = A_full.tocsr()
rows_A, cols_A = A_full.nonzero()
vals_A = A_full.data
############# call Mosek #############
from mosek import *
x_all = [0.0] * (n*m + 3*n)
env = Env()
task = env.Task(0, 1)
task.putintparam(iparam.log, 10)
task.putintparam(iparam.intpnt_multi_thread, onoffkey.off)
# task.putdouparam(dparam.intpnt_tol_rel_gap, 1e-8)
# variables
task.appendvars(n*m + 3*n) # in the order x[i,j], u[i], s[i], t[i]
task.putvarboundlist(list(np.arange(n*m+3*n)), [boundkey.lo] * (n*m) + [boundkey.fr] * (3*n), [0.0]*(n*m+3*n), [100]*(n*m+3*n))
# obj.
c_vec = np.zeros((n*m+3*n,))
c_vec[-n:] = -B
task.putclist(np.arange(n*m+3*n), c_vec)
task.putobjsense(objsense.minimize)
# constr.
task.appendcons(2*n+m)
task.putaijlist(rows_A, cols_A, vals_A)
rhs = list(np.concatenate([np.zeros((n,)), np.ones(n + m)]))
task.putconboundslice(0, 2*n+m, [boundkey.fx]*(2*n+m), rhs, rhs)
[task.appendcone(conetype.pexp, 0.0, [ii, ii+n, ii+2*n]) for ii in range(n*m, n*m+n)]
begin_msk = time()
task.optimize()
task.getsolsta(soltype.itr)
task.getxx(soltype.itr, x_all)
x_msk = np.reshape(x_all[:n*m], (n, m))
msk_direct_obj = task.getprimalobj(soltype.itr)
b_msk = compute_b_from_x(x_msk) # np.sum(b_msk, 1) - B is very small
ave_dgap_msk = compute_ave_dgap(b_msk)
time_msk = time() - begin_msk
print("Mosek (direct) time = {}, ave_dgap = {}".format(time_msk, ave_dgap_msk))
print("||np.sum(x, 0)-1|| = {}".format(np.linalg.norm(np.sum(x_msk, 0)-1)))

########################################################################################################################
print("======== proximal gradient on the EG primal (linear) with backtracking linesearch ========")
# @jit(nopython=True)
x = np.multiply(B/sum_B, np.ones(shape=(m,n)), dtype=np.float32).T
obj_curr, gg = f_and_grad_eg(x)
def pgls_for_eg(v, B):
    # u_min = B/sum_B * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on z[i] = v[i].dot(x[i])
    x = np.multiply(B/sum_B, np.ones(shape=(m,n)), dtype=np.float32).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
    step = 100/Lf_eg # initial large stepsize
    max_ls = 20 # max number of backtracking steps in linesearch
    bt_fac = 0.5 # backtracking discount factor
    inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
    total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
    # total_bt_list = []
    ind = np.arange(1, n+1)
    u = np.zeros((n, m))
    temp_total_time = 0
    for iter_idx in range(1, max_iter + 1):
        obj_curr, gg = f_and_grad_eg(x)
        # obj_curr, gg = f_eg(x), grad_f_eg(x) # gradient
        for ls_idx in range(1, max_ls + 1):
            # tt_proj = time()
            x_bar = x - step * gg
            # sorted_indices = np.argsort(x_bar, axis=0)
            temp_begin_time = time()
            u = np.sort(x_bar, axis=0)[::-1]
            temp_total_time += time() - temp_begin_time
            cssv = np.cumsum(u, axis=0) - 1 # cssv[:, j] = np.cumsum(u[:, j]) - s[j]
            # cond = u - (cssv.T/ind).T > 0
            pivot_indices = np.argmin(u - (cssv.T/ind).T > 0, axis=0) - 1
            theta_vec = cssv[pivot_indices, range(m)] / ind[pivot_indices]
            x_try, p_try = np.maximum(x_bar - theta_vec, 0), theta_vec/step
            if step <= 1/Lf_eg: # already small step
                break
            obj_try = f_eg(x_try)
            f_hat_try = obj_curr + np.sum(gg * (x_try - x)) + 1/(2*step) * np.sum((x_try - x)**2) # pseudo quad. approx. 
            if obj_try <= f_hat_try:
                break # brek if sufficient decrease
            step *= bt_fac # otherwise decrease step
        x, p = x_try, p_try
        total_bt += ls_idx # update total number of backtracking (minimum is 1)
        # total_bt_list.append(total_bt)
        if ls_idx == 1: # increase step if no backtracking is performed
            step *= inc_fac # print("increase step to {}".format(step))
        if iter_idx % max(max_iter//200, 1) == 0:
            # print("density of x_bar = {}".format(np.sum(x_bar >= 1e-4)/(n*m)))
            primal_eg_obj = -np.sum(B * np.log(np.sum(v * x, axis=1)))
            beta = np.min(p/v, axis = 1)
            dual_eg_obj = np.sum(B * np.log(beta))
            ave_dgap = (primal_eg_obj - dual_eg_obj)/n
            print("PGLS iter = {}, total ls. = {}, ave_dgap = {:.5f}".format(iter_idx, total_bt, ave_dgap))
            if ave_dgap <= accu:
                print("PGLS ave_dgap = {:.5f} <= {} at iter = {}, ls = {}".format(ave_dgap, accu, iter_idx, total_bt))
                break
    print("sort total time = {}".format(temp_total_time)) 

begin = time()
pgls_for_eg(v, B)
time_pg = time() - begin
print("time elapsed = {}".format(time_pg))

##################################################################
print("======== Proportional Response (Wu & Zhang 2007, Zhang 2009, Birnbaum et al. 2011) ========")
begin = time()
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # initial (uniform) allocation, t = 0
# bids b[i,j] = x[i,j] * v[i,j]
b = v * x
# dgap_array, ave_dgap_array = [], []
for iter_idx in range(1, max_iter+1):
    ###############  ###############
    p = np.sum(b, axis=0) # compute prices
    x = b / p # new allocation
    b = compute_b_from_x(x) # new bids
    # compute residuals and print
    if (iter_idx % (max_iter//200)) == 0:
        ave_dgap = compute_ave_dgap(b)
        print("PR iter = {}, ave_dgap = {:.5f}".format(iter_idx, ave_dgap))
    # dgap_array.append(dgap), 
    # ave_dgap_array.append(ave_dgap)
        if ave_dgap <= accu: # if dgap <= accu: # no need to go further...
            print("PR ave_dgap <= {} reached at iter = {}".format(accu, iter_idx))
            break
time_pr = time() - begin
print("time elapsed = {}".format(time_pr))

begin = time()
##################################################################
do_ls_fw = False # for FW algorithm
print("======== original Frank-Wolfe on EG (linesearch: {}) ========".format(do_ls_fw))
x = np.zeros(shape = (n, m)) # random initialization
x[range(n), range(n)] = 1
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
    if iter_idx % (max_iter//200) == 0:
        b_fw = compute_b_from_x(x)
        ub_fw = compute_ave_dgap(b_fw)
        print("FWLS iter = {}, exact ls. = {}, ave_dgap = {:.5f}".format(iter_idx, count_exact_ls, ub_fw))
        if ub_fw <= accu:
            print("FWLS ub_fw <= {} at iter {}, break".format(accu, iter_idx))
            break

time_fw = time() - begin
print("time elapsed = {}".format(time_fw))

csv_writer.writerow([sd, time_msk, time_pg, time_pr, time_fw])
ff.close() # close file