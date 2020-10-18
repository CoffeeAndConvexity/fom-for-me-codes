from utils import * 
import cvxpy as cp
from time import time
from scipy.optimize import bisect
import os, csv, argparse

# arguments for experiment setups
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
    parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
    parser.add_argument('--m', type=int, default=100, help='m = number of items')
    parser.add_argument('--max_iter', type=int, default=20000, help='max number of iterations')
    parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
    parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
    parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-3, help='stop early when dgap or p_diff <= da')
    args = parser.parse_args()
    # retrieve arguments
    n, m = args.n, args.m
    distr_name = args.distr_name
    file_mode = args.file_mode
    max_iter = args.max_iter
    sd = args.seed
    accu = args.desired_accuracy
except:
    # sd = 2, n = 200, m = 400, log_normal num. ls. = 21672.0
    print("Error parsing arguments. Use default ones.")
    n, m, distr_name, file_mode, max_iter, sd, accu = 200, 400, 'lognormal', 'w', 20000, 2, 1e-3

varying_B = True

# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join("logs/leontief_varying_B", fname) if varying_B else os.path.join("logs/leontief_B_1", fname)
try:
    print("file mode (w or a): {}".format(file_mode))
    ff = open(fpath, file_mode)
except:
    ff = open(fpath, 'w')
csv_writer = csv.writer(ff)

np.random.seed(sd)

# distribution for random matrix generation
if distr_name == "unif":
    distr = np.random.uniform # np.random.exponential
elif distr_name == "exp":
    distr = np.random.exponential
elif distr_name == 'normal': 
    distr = np.random.normal
elif distr_name == 'log_normal' or distr_name == 'lognormal':
    distr = np.random.lognormal

a = np.abs(distr(size = (n, m))) # buyers' valuations
# just subsample some of them
num_zeros = int(n * m * 0.5)
a[np.random.randint(n, size = num_zeros), np.random.randint(m, size = num_zeros)] = 0
print("density of a = {}".format(np.sum(a!=0)/(n*m)))
B = np.abs(distr(size = (n,))) + 0.5 if varying_B else np.ones(shape = (n,))
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

################ useful constants ################
r_low = np.max(a, 1) * B # lower bounds on <a[i], p>, where a[i] = (a[i,j])
sum_B = np.sum(B)
B_log_r_low = B * np.log(r_low)
B_over_r_low = B/r_low
B_over_r_low_sq = B/(r_low**2)
# r_high = np.sum(B) * np.max(a, 1)
L = np.max(B/r_low) # Lipschitz constant of h_tilde
Lf = L * np.linalg.norm(a, ord=2) # np.linalg.svd(a, full_matrices=False, compute_uv=False)[0]

def f(p): # the original objective function
    return -B @ (np.log(a@p))

def f_extrap(p):
    rr = a@p
    above_mask, below_mask = (rr >= r_low), (rr<r_low)
    term_above = -B[above_mask] * np.log(rr[above_mask])
    term_below = -B_log_r_low[below_mask] - B_over_r_low[below_mask] * (rr[below_mask] - r_low[below_mask]) + 0.5 * B_over_r_low_sq[below_mask] * (rr[below_mask] - r_low[below_mask])**2
    return np.sum(term_above) + np.sum(term_below)
    # rr = np.maximum(a@p, r_low)
    # return -B.dot(rr)

def grad_f(p):
    return -np.sum(((a.T*B)/(a@p)).T, axis=0)

def grad_f_extrap(p): # when r[i] = <a[i], p> is small, replace it with a quadratic
    rr = a@p
    above_mask, below_mask = (rr >= r_low), (rr<r_low)
    grad_above = -(B[above_mask]/rr[above_mask]) @ a[above_mask]
    zz = -B_over_r_low[below_mask] + B_over_r_low_sq[below_mask]*(rr[below_mask]-r_low[below_mask])
    grad_below = zz @ a[below_mask]
    return grad_above + grad_below

def compute_u_from_p(p):
    u = B/(a@p) # by corr. primal variables by KKT stationarity condition
    return u / np.max(a.T@u) # normalize since it might be infeasible

def duality_gap(p): # duality gap using the original functions
    rr = a@p
    u = B/rr
    u = u / np.max(a.T@u) # normalize
    return - B @ np.log(u) - B @ np.log(rr)

print("======== solve the dual in p using CVXPY + Mosek ========")
begin = time() # time it
p_cp = cp.Variable(shape=(m,), nonneg=True)
obj_expr = - cp.sum( B * cp.log( a@ p_cp ))
# for i in range(n):
#     obj_expr -= B[i] * cp.log(a[i] @ p_cp)
objective = cp.Minimize(obj_expr)
constraints = [cp.sum(p_cp) == sum_B] # linear constraints
prob = cp.Problem(objective, constraints) # define the optimization problem
cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True) # solve it using SCS
p_cp = p_cp.value
print("cvxpy (dual) opt_obj = {}, time elapsed = {}".format(f(p_cp), time()-begin))
u_cp = compute_u_from_p(p_cp)

# duality_gap(p_cp)
# duality_gap(p_pg)

# print("======== solve the primal in u using CVXPY + Mosek ========")
# begin = time() # time it
# u_cp = cp.Variable(shape=(n,), nonneg=True)
# obj_expr = 0
# obj_expr = - cp.sum(B * cp.log(u_cp))
# objective = cp.Minimize(obj_expr)
# constraints = [a.T @ u_cp <= 1] # linear constraints
# prob = cp.Problem(objective, constraints) # define the optimization problem
# cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True) # solve it using SCS
# u_cp = u_cp.value
# print("cvxpy (primal) opt_obj = {}, time elapsed = {}".format(cvxpy_opt_obj, time()-begin))

print("====================== PGLS ======================")
step = 0.5
step = 1000/Lf
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
u_diff_array = []
p_pg = sum_B * np.ones((m,))/m
for iter_idx in range(1, max_iter+1):
    obj_curr, gg = f_extrap(p_pg), grad_f_extrap(p_pg) # curr. obj and grad
    for ls_idx in range(1, max_ls+1):
        p_try = proj_simplex(p_pg - step * gg, s=sum_B)
        if step <= 1/Lf: # small step, break
            break
        obj_try = f_extrap(p_try)
        f_hat_try = obj_curr + gg @ (p_try - p_pg) + 1/(2*step) * sum_squares(p_try-p_pg)
        if obj_try <= f_hat_try: # sufficient decrease step, break 
            break
        step *= bt_fac # backtracking
    p_pg = p_try # update iterate
    total_bt += ls_idx
    total_bt_list.append(total_bt)
    if ls_idx == 1: # if no backtracking occurs, increase curr. step
        step *= inc_fac
    u_diff = np.max(np.abs(compute_u_from_p(p_pg) - u_cp)/u_cp)
    u_diff_array.append(u_diff)
    if iter_idx % (max_iter//10) == 0:
        # u_diff = np.linalg.norm(compute_u_from_p(p_cp) - compute_u_from_p(p_pg))
        print("sd = {}, distr = {}, iter = {}, ls = {}, u_diff = {}".format(sd, distr_name, iter_idx, total_bt, u_diff))
    if u_diff <= accu:
        print("iter = {}, ls = {}, u_diff = {} <= {}, break".format(iter_idx, total_bt, u_diff, accu))
        break

rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pg'] * max_iter, range(1, max_iter+1), u_diff_array, total_bt_list)
csv_writer.writerows(rows)

# do_ls_fw = False
# print("====================== FW (linesearch: {}) ======================".format(do_ls_fw))
# # random vertex initialization
# # p_fw = np.zeros((m,))
# # p_fw[np.random.randint(m)] = 1
# p_fw = sum_B * np.ones((m,))/m
# ave_dgap_array = []
# count_exact_ls = 0
# for iter_idx in range(1, max_iter+1):
#     vertex_fw = np.argmin(grad_f_extrap(p_fw)) # vertex vv that minimizes <grad, vv>, one-hot vector
#     # construct direction dd = vv - p_fw
#     dd = -p_fw
#     dd[vertex_fw] += sum_B
#     if do_ls_fw:
#         # one-dim function in stepsize
#         one_dim_func = (lambda ll: grad_f_extrap(p_fw + ll * dd) @ dd) # gradient w.r.t. stepsize
#         if one_dim_func(0) > 0: # should not happen
#             print("Warning: FW exact linesearch gives gamma = 0")
#             gamma = 0
#         elif one_dim_func(1) < 0:
#             print("Warning: FW use gamma_max")
#             gamma = 1
#         else: # minimizer is in between
#             count_exact_ls += 1
#             gamma = bisect(one_dim_func, 0, 1, disp=False)
#         # p_new = p_curr + gamma * dd
#     else:
#         gamma = 2/(iter_idx + 2)
#     p_fw = (1 - gamma)*p_fw
#     p_fw[vertex_fw] += gamma * sum_B
#     ave_dgap = duality_gap(p_fw)/n
#     ave_dgap_array.append(ave_dgap)
#     if iter_idx % (max_iter//10) == 0:
#         print("iter = {}, exact ls = {}, obj = {:.5f}, dgap/n = {:.5f}".format(iter_idx, count_exact_ls, f(p_fw), ave_dgap))
#     if ave_dgap <= accu:
#         print("iter = {}, exact ls = {}, obj = {:.5f}, dgap/n = {:.5f} <= {}, break".format(iter_idx, count_exact_ls, f(p_fw), ave_dgap, accu))
#         break

# rows = zip([sd]*max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['fwls'] * max_iter, range(1, max_iter+1), ave_dgap_array)
# csv_writer.writerows(rows)

# close file
ff.close()