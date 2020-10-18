""" load module, define constants and functions"""

from utils import * 
import cvxpy as cp
from time import time
import os, csv, argparse
from scipy.optimize import bisect

#######################################################################################
# arguments for experiment setups
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
fpath = os.path.join("logs/linear_varying_B", fname)
try:
    print("file mode (w or a): {}".format(file_mode))
    ff = open(fpath, file_mode)
except:
    ff = open(fpath, 'w')
csv_writer = csv.writer(ff)

print("n = {}, m = {}, seed = {}, distr. of v = {}".format(n, m, sd, distr_name))

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
# B = np.ones(shape = (n,)) # all same budget
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies

##################################################################
# some useful constants and functions
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on u[i] = v[i].dot(x[i])
L = np.max(B/u_min**2) # Lipschitz constant of sum of -B[i]*log(z[i])
# Lf_eg = L * np.linalg.norm(v, ord='fro') ** 2 # Lipschitz constant of EG f(x) = - sum B[i]*log(v[i].dot(x[i]))
Lf_eg = L * np.max(np.linalg.norm(v, axis=1))**2
# objective function # z = np.random.uniform(size=(n,))
neg_B_log_u_min = - B * np.log(u_min)
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

############ not used currently ############
p_upper_linear = sum_B # any optimal p[j] must <= total budget
denom_vec = np.sum(v, axis = 1)
# p_lower = np.array([np.max(v[:, j] * B / denom_vec) for j in range(m)])
temp_mat = (v.T * B / denom_vec).T # temp_mat[2, 3] == v[2,3]*B[2]/denom_vec[2]
p_lower_linear = np.max(temp_mat, axis = 0)
# Lipschitz constant for gradient, Lf = Lh * sig_max(A.T@A)
Lf_linear_shmyrev = 1/np.min(p_lower_linear) * (n * m)

def h_eg(z): # sum of -B[i] * log(z[i]) with extrap.
    # is_above = (z >= u_min)
    above_array = -B * np.log(np.maximum(z, u_min))
    below_array = 0.5 * B/(u_min**2) * (z - u_min)**2 + (-B/u_min) * (z - u_min) + neg_B_log_u_min
    return np.sum(above_array[z >= u_min]) + np.sum(below_array[z<u_min])

def f_eg(x): # actual EG obj. in optimization (with quad. extrapolation)
    return h_eg(np.sum(v * x, 1))

def primal_obj_eg(x): # original obj. assuming sum(x[:, j]) == s[j] for all j
    return -np.sum(B * np.log(np.sum(v * x, axis=1)))

def grad_f_eg(x):
    """ compute z[i] = v[i].dot(x[i]) and gradient """
    z = np.sum(v * x, 1) # z = np.maximum(np.sum(v * x, 1), u_min)
    gh = - B/np.maximum(z, u_min) * (z>=u_min) + B/(u_min**2)*(z-2*u_min) * (z<u_min)
    return (v.T*gh).T
    # for i in range(n):
    #     # ghi = - B[i]/z[i] if z[i] >= u_min[i] else B[i]/(u_min[i]**2) * (z[i] - 2*u_min[i]) # derivative of quad. extrapolation
    #     ghi = gh[i]
    #     gg[i, :] = ghi * v[i, :] # grad w.r.t. x[i], by chain rule
    # return gg

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

def compute_b_from_x(x):
    tt = v*x
    return ((tt.T / np.sum(tt, 1)) * B).T

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

# print("======== proximal gradient on the primal with static stepsizes ========")
# step = 1/Lf_eg
# x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T
# p_diff_array = []
# for iter_idx in range(1, max_iter + 1):
#     x, p = proj_simplex_all(x - step * grad_f_eg(x), return_multipliers=True)
#     p = np.sum(compute_b_from_x(x), 0)
#     p_diff = np.max(np.abs(p - p_cp)/p_cp)
#     if iter_idx % max(max_iter//10, 1) == 0:
#         obj_val = f_eg(x)
#         print("iter = {}, obj = {:.5f}, p_diff = {:.5f}".format(iter_idx, obj_val, p_diff))   

print("======== proximal gradient on the primal with backtracking linesearch ========")
u_min = B/np.sum(B) * np.sum(v, axis=1) # the proportionality lower bound u_min[i] on z[i] = v[i].dot(x[i])
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # x = np.array([np.ones(shape=(m, )) * B[i] for i in range(n)]) / np.sum(B)
p_diff = []
# step = 100/Lf_eg # initial large stepsize 1000/Lf_eg when Lf_eg is wrong
step = 100/Lf_eg
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx in range(1, max_iter + 1):
    gg = grad_f_eg(x) # gradient
    obj_curr = f_eg(x)
    for ls_idx in range(1, max_ls + 1):
        x_try = proj_simplex_all(x - step * gg, return_multipliers=False)
        if step <= 1/Lf_eg: # break if already very small step
            break # since safe step is very small, there can be numerical issues that lead to furtuer backtracking and make it extremely small
        obj_try = f_eg(x_try)
        f_hat_try = obj_curr + np.sum(gg * (x_try - x)) + 1/(2*step) * sum_squares(x_try - x) # pseudo quad. approx. 
        if obj_try <= f_hat_try: # brek if sufficient decrease
            break
        # otherwise decrease step
        step *= bt_fac
    # x, p = x_try, mult_try / step # update x and p
    x = x_try
    p = np.sum(compute_b_from_x(x), 0)
    # p = mult_try / step
    total_bt += ls_idx # update total number of backtracking (minimum is 1)
    # print("iter = {}, num. of ls = {}, step = {}".format(iter_idx, ls_idx, step))
    total_bt_list.append(total_bt)
    if ls_idx == 1: # increase step if no backtracking is performed
        step *= inc_fac
        # print("increase step to {}".format(step))
    rel_diff_p_pg = np.max(np.abs(p-p_cp)/p_cp) # relative difference in price
    if iter_idx % max(max_iter//10, 1) == 0:
        print(eg_duality_gap(x, p))
        obj_val = f_eg(x)
        # rel_diff_x = np.linalg.norm(x - x_cp.value)/np.linalg.norm(x_cp.value) # diff. from the cvxpy optimal solution
        print("iter = {}, obj = {:.5f}, num. bt. ls. = {}, max(np.abs(p-p_cp)/p_cp) = {:.5f}".format(iter_idx, obj_val, total_bt, rel_diff_p_pg))
    # compute p_diff[t] and append to list
    p_diff.append(rel_diff_p_pg)
    if rel_diff_p_pg <= accu:
        print("accu. reached at iter = {}, ls = {}".format(iter_idx, total_bt))
        break

# save to file; both iter & ls counts are saved
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls'] * max_iter, range(1, max_iter + 1), p_diff, total_bt_list)
csv_writer.writerows(rows)

print("======== Proportional Response (Wu & Zhang 2007, Zhang 2009, Birnbaum et al. 2011) ========")
begin = time()
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # initial (uniform) allocation, t = 0
# bids b[i,j] = x[i,j] * v[i,j]
b = v * x # t = 1
p_diff_array = []
for iter_idx in range(1, max_iter+1):
    p = np.sum(b, axis=0) 
    x = b / p
    tt = v*x
    b = ((tt.T / np.sum(tt, 1)) * B).T
    # compare prices
    rel_diff_p_pr = max(np.abs(p-p_cp)/p_cp)
    if (iter_idx % (max_iter//10)) == 0:
        obj_val = f_linear_shmyrev(x)
        # rel_diff_x = np.linalg.norm(x - x_cp.value)/np.linalg.norm(x_cp.value)
        print("iter = {}, obj = {:.5f}, rel_diff_p_pr = {:.5f}".format(iter_idx, obj_val, rel_diff_p_pr))
    p_diff_array.append(rel_diff_p_pr)
    # if dgap <= accu: # no need to go further...
    if rel_diff_p_pr <= accu:
        print("accu. reached at iter = {}".format(iter_idx))
        break

time_elapased = time() - begin
print("time elapsed = {}".format(time_elapased))

#######################################################################################################################
# write to file and close (max_iter is 4000, but may have less than 4000 rows since stopping early when dgap <= 1e-5)
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr'] * max_iter, range(1, max_iter+1), p_diff_array)
csv_writer.writerows(rows)

##################################################################
do_ls_fw = True
print("======== original Frank-Wolfe on EG (linesearch: {}) ========".format(do_ls_fw))
x = np.zeros((n, m))
x[range(n), range(n)] = 1
p_diff_array = []
count_exact_ls = 0
for iter_idx in range(1, max_iter+1):
    # compute gradient w.r.t. x
    gg = grad_f_eg(x)
    # find the vertex that minimizes the gradient
    vertex_fw = tuple(np.argmin(gg, axis=0)) # x[i,j] = 1 if i = x_fw_indices[j] and = 0 o.w.
    # find stepsize
    if do_ls_fw:
        vx = np.sum(v*x, axis=1)
        # dd = vertex_to_mat(vertex_fw) - x # the following two lines do the same
        dd = -x
        dd[vertex_fw, range(m)] += 1
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
    # compute residual measure
    # b_fw = compute_b_from_x(x)
    p_fw = np.sum(compute_b_from_x(x), 0)
    p_diff_fw = np.max(np.abs(p_fw - p_cp) / p_cp)
    p_diff_array.append(p_diff_fw)
    if iter_idx % (max_iter//10) == 0:
        print("iter = {}, exact ls. = {}, p_diff = {:.5f}".format(iter_idx, count_exact_ls, p_diff_fw))
    if p_diff_fw <= accu:
        print("ub_fw <= {} at iter {}, break".format(accu, iter_idx))
        break

# save to file
rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['fwls'] * max_iter, range(1, max_iter+1), p_diff_array)
csv_writer.writerows(rows)

ff.close()