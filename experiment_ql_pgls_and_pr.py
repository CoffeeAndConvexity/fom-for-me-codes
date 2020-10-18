# trying out PGLS and PR (MD) for QL-Shmyrev

""" load module, define constants and functions"""
from utils import * 
import cvxpy as cp
from time import time
import os, csv, argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str, default='logs/ql/', help='log file mode (w or a)')
parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
parser.add_argument('--m', type=int, default=100, help='m = number of items')
parser.add_argument('--max_iter', type=int, default=50000, help='max number of iterations')
parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-3, help='stop early when dgap <= da')
args = parser.parse_args()
# retrieve arguments
n, m = args.n, args.m
# log_path = args.path
distr_name = args.distr_name
file_mode = args.file_mode
max_iter = args.max_iter
sd = args.seed
accu = args.desired_accuracy

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
# B = np.abs(distr(size = (n,))) + 0.5 # buyers' valuations
B = np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets
# assume s[i] = 1 for all i

# useful functions and constants
# lower and upper bounds on p(i)
p_upper = np.max(v, axis = 0)
denom_vec = np.sum(v, axis = 1) + B
# p_lower = np.array([np.max(v[:, j] * B / denom_vec) for j in range(m)])
temp_mat = (v.T * B / denom_vec).T # temp_mat[2, 3] == v[2,3]*B[2]/denom_vec[2]
p_lower = np.max(temp_mat, axis = 0)

def f(b):
    ''' the QL-Shmyrev minimization objective in bids b = (b[i,j])'''
    pb = np.sum(b, axis = 0)
    return -np.sum((1 + np.log(v)) * b) + np.sum(pb * np.log(pb))

def grad_f(b): # gradient of f(b) at any b > 0
    """ compute grad[i,j] = p[j]/b[i,j], where p[j] = sum of b[i,j] over i """
    return np.log(np.sum(b, axis = 0)/v)

def duality_gap(p, b): # given p and b, compute duality gap
    # obj. of the dual in terms of (p, beta)
    beta = np.min(p/v, 1)
    obj_dual = np.sum(p) - np.sum(B * np.log(beta))
    obj_primal = f(b)
    return obj_dual + obj_primal # - np.sum(B * (1 - np.log(B)))

# upper and lower bounds
p_upper = np.max(v, axis=0)
p_lower = np.max((v.T * np.exp(-np.sum(p_upper)/B)), axis=1) # p_lower[j] = max_i v[i,j] * np.exp(-np.sum(p_upper)/B[i])
p_lower = np.maximum(p_lower, 1e-1)
# Lipschitz constant for gradient, Lf = Lh * sig_max(A.T@A)
Lf = 1/np.min(p_lower) * (n * m)

# print("======== solve QL-EG using Mosek ========")
# # Cole et al. (2016) Lemma 5
# begin = time() # time it
# x_cp = cp.Variable(shape = (n, m), nonneg=True) # allocations
# w_cp = cp.Variable(shape = (n,), nonneg=True) # "surplus"
# u_cp = cp.Variable(shape = (n,), nonneg=True) # utilities - can be eleminated
# obj_expr = - cp.sum(B * cp.log(u_cp)) + cp.sum(w_cp) - np.sum(B * (1 - np.log(B))) # keep the constant to match the other objectives
# objective = cp.Minimize(obj_expr)
# constraints = [u_cp[i] <= (v[i] @ x_cp[i] + w_cp[i]) for i in range(n)] # utilities <= value from obtained goods and remaining budget
# constraints.extend([cp.sum(x_cp[:, j]) <= 1 for j in range(m)]) # 
# prob = cp.Problem(objective, constraints) # define the optimization problem
# cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True) # solve it using SCS
# prices_via_eg = np.array([constraints[j].dual_value for j in range(n, n+m)])
# time_elapased = time() - begin
# print("min obj = {}, time elpased = {}".format(cvxpy_opt_obj, time_elapased))

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
# rel_diff_prices = np.max(np.abs((prices_via_eg - price_via_shmyrev_mosek)/price_via_shmyrev_mosek))
# print("diff. between QL-EG-Mosek and QL-Shmyrev-Mosek prices = {}".format(rel_diff_prices))
# print('prices = ', p_cp.value)

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
    dgap_pr = duality_gap(p_pr, b_pr)
    dgap_array_pr.append(dgap_pr)
    # print(dgap_pr)
    if p_diff_pr <= accu:
        print("p_diff <= {} at iter = {}, break".format(accu, iter_idx_pr))
        break
    if iter_idx_pr % (max_iter//5) == 0:
        print("iter = {}, obj val. = {}, p_diff = {}".format(iter_idx_pr, f(b_pr), p_diff_pr))
    # for i in range(n): # the old way
    #     denom = (np.sum(tt[i]) + delta[i])
    #     b[i] = B[i] * tt[i] / denom
    #     delta[i] = B[i] * delta[i] / denom
time_elapased = time() - begin
print("time elpased = {}".format(time_elapased))
rel_diff_prices = np.max(np.abs((p_pr - price_via_shmyrev_mosek)/price_via_shmyrev_mosek))

# write to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr'] * max_iter, range(1, max_iter + 1), p_diff_array_pr, dgap_array_pr)
csv_writer.writerows(rows)

# print("diff. between QL-Shmyrev-Mosek and QL-Shmyrev-PR prices = {}".format(rel_diff_prices))

# print("======== solve QL-Shmyrev using fixed step PG ========")
# max_iter_pg = 1000
# b_aug = np.outer(np.ones(m+1, ), B/(m+1)).T # b_aug = (b, delta)
# g_aug = np.zeros((n, m+1)) # to store gradient
# step = 1000/Lf # a fixed stepsize
# for iter_idx in range(1, max_iter_pg+1):
#     g_aug[:, :m] = grad_f(b_aug[:, :m]) # compute grad. vec. of the b part (grad. for the delta part is 0)
#     for i in range(n): # projected gradient step
#         b_aug[i, :] = proj_simplex(b_aug[i, :] - step * g_aug[i, :], s=B[i])
#     if iter_idx % (max_iter_pg//5) == 0:
#         print("iter = {}, obj val. = {}".format(iter_idx, f(b_aug[:, :m])))

print("======== solve QL-Shmyrev using PGLS ========")
p_diff_array_pgls = []
dgap_array_pgls = []
b_aug = np.outer(np.ones(m+1, ), B/(m+1)).T # b_aug = (b, delta)
g_aug = np.zeros((n, m+1)) # to store gradient
step = 1000/Lf # start from a large stepsize
# dgap_array = []
max_ls = 20 # max number of backtracking steps in linesearch
bt_fac = 0.8 # backtracking discount factor
inc_fac = 1.02 # factor for increasing the stepsize (for the next iteration) if no backtracking
total_bt = 0 # every iteration has bt >= 1 (i.e., bt == 1 means NO actual backtracking occurs)
total_bt_list = []
for iter_idx_pgls in range(1, max_iter+1):
    g_aug[:, :m] = grad_f(b_aug[:, :m]) # compute grad. vec. of the b part (grad. for the delta part is 0)
    obj_curr = f(b_aug[:, :m]) # compute current objective
    for ls_idx in range(1, max_ls + 1): # linesearch iterations
        b_aug_try = np.stack([proj_simplex(b_aug[i, :] - step * g_aug[i, :], s=B[i]) for i in range(n)])
        if step <= 1/Lf:
            break
        obj_try = f(b_aug_try[:, :m])
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
    b_pgls = b_aug[:, :m]
    p_diff_pgls = np.max(np.abs(p_pgls-price_via_shmyrev_mosek)/price_via_shmyrev_mosek) # relative difference in price
    dgap_pgls = duality_gap(p_pgls, b_pgls)
    dgap_array_pgls.append(dgap_pgls)
    p_diff_array_pgls.append(p_diff_pgls)
    if p_diff_pgls <= accu:
        print("p_diff <= {} at iter = {}, ls = {}, break".format(accu, iter_idx_pgls, total_bt))
        break
    p_diff_array_pgls.append(p_diff_pgls)
    if iter_idx_pgls % (max_iter//5) == 0:
        print("iter = {}, total_ls = {}, obj val. = {}, rel_diff_p = {}".format(iter_idx_pgls, total_bt, f(b_aug[:, :m]), p_diff_pgls))

# write running logs to file
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls'] * max_iter, range(1, max_iter + 1), p_diff_array_pgls, dgap_array_pgls, total_bt_list)
csv_writer.writerows(rows)
ff.close()