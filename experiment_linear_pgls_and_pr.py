#############################################################################
#############################################################################
#############################################################################
""" load module, define constants and functions"""

from utils import * 
import cvxpy as cp
from time import time
import os, csv, argparse

#######################################################################################
# arguments for experiment setups
parser = argparse.ArgumentParser()
parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
parser.add_argument('--m', type=int, default=100, help='m = number of items')
parser.add_argument('--max_iter', type=int, default=50000, help='max number of iterations')
parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-2, help='stop early when dgap or p_diff <= da')
args = parser.parse_args()
# retrieve arguments
n, m = args.n, args.m
distr_name = args.distr_name
file_mode = args.file_mode
max_iter = args.max_iter
sd = args.seed
accu = args.desired_accuracy

np.random.seed(sd)

###################################################################
# log file writing misc
fname = "sd-{}-n-{}.csv".format(sd, n) # log file name
fpath = os.path.join("logs/linear", fname)
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
# B = np.abs(distr(size = (n,))) + 0.5 # buyers' budgets
B = np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies

##################################################################
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

print("======== proximal gradient on the primal with backtracking linesearch ========")
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
        print("iter = {}, obj = {:.5f}, total num. of backtracking = {}, duality gap = {:.5f}, max(np.abs(p-p_cp)/p_cp) = {:.5f}".format(iter_idx, obj_val, total_bt, dgap, rel_diff_p))
    # compute p_diff[t] and append to list
    p_diff.append(rel_diff_p)
    dgap_array.append(dgap)
    # if dgap <= accu: # no need to go further; otherwise may lead to numerical issues
    if rel_diff_p <= accu:
        print("accu. reached at iter = {}, ls = {}".format(iter_idx, total_bt))
        break

# save to file; both iter & ls counts are saved
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pgls'] * max_iter, range(1, max_iter + 1), p_diff, dgap_array, total_bt_list)
csv_writer.writerows(rows)

print("======== Proportional Response (Wu & Zhang 2007, Zhang 2009, Birnbaum et al. 2011) ========")
begin = time()
x = np.multiply((B/sum(B)), np.ones(shape=(m,n))).T # initial (uniform) allocation, t = 0
# bids b[i,j] = x[i,j] * v[i,j]
b = v * x # t = 1
p_diff_array = []
dgap_array = []
for iter_idx in range(1, max_iter+1):
    ############### compute prices ###############
    p = np.sum(b, axis=0) 
    ############### compute new allocation ###############
    x = b / p
    # print(np.linalg.norm(np.sum(x, 0) - s))
    # for j in range(m):
    #     x[:,j] = b[:, j] / p[j]
    ############### compute new bids ###############
    tt = v*x
    b = ((tt.T / np.sum(tt, 1)) * B).T
    # for i in range(n):
    #     vixi = v[i] * x[i]
    #     b[i, :] = B[i] * (vixi/np.sum(vixi))
    ############### compare current p and p_opt ###############
    rel_diff_p = max(np.abs(p-p_cp)/p_cp)
    dgap = eg_duality_gap(x, p)
    if (iter_idx % (max_iter//10)) == 0:
        obj_val = f(x)
        # rel_diff_x = np.linalg.norm(x - x_cp.value)/np.linalg.norm(x_cp.value)
        print("iter = {}, obj = {:.5f}, duality gap = {:.5f}, max(np.abs(p-p_cp)/p_cp) = {:.5f}".format(iter_idx, obj_val, dgap, rel_diff_p))
    p_diff_array.append(rel_diff_p)
    dgap_array.append(dgap)
    # if dgap <= accu: # no need to go further...
    if rel_diff_p <= accu:
        print("accu. reached at iter = {}".format(iter_idx))
        break

time_elapased = time() - begin
print("time elapsed = {}".format(time_elapased))

#######################################################################################################################
# write to file and close (max_iter is 4000, but may have less than 4000 rows since stopping early when dgap <= 1e-5)
rows = zip([sd] * max_iter, [n]*max_iter, [m]*max_iter, [distr_name] * max_iter, ['pr'] * max_iter, range(1, max_iter+1), p_diff_array, dgap_array)
csv_writer.writerows(rows)
ff.close()