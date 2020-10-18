from utils import * 
import cvxpy as cp
from time import time
import os, csv, argparse

# arguments for experiment setups
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_mode', '--fm', type=str, default='w', help='log file mode (w or a)')
    parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
    parser.add_argument('--m', type=int, default=100, help='m = number of items')
    parser.add_argument('--max_iter', type=int, default=50000, help='max number of iterations')
    parser.add_argument('--distr_name', '--dn', type=str, default='unif', help='data generation distibution name (unif, normal, exp, log_normal)')
    parser.add_argument('--seed', '--sd', type=int, default=1, help='random seed')
    parser.add_argument('--desired_accuracy', '--da', type=float, default=1e-5, help='stop early when dgap or p_diff <= da')
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
    n, m, distr_name, file_mode, max_iter, sd, accu = 10, 20, 'unif', 'w', 5000, 1, 1e-3

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
# B = np.abs(distr(size = (n,))) + 0.5 # buyers' budgets
B = np.ones(shape = (n,)) # buyers' budgets # B = np.random.exponential(scale = 1.5, size = (n,)) # buyers' budgets
s = np.ones(shape = (m,)) # unit supplies # s = np.random.exponential(scale = 2, size = (m, )) # sellers' supplies
sum_B_log_B = np.sum(B * np.log(B))
sum_B = np.sum(B)

def f(p): # the objective function
    return -np.sum(B * np.log(a@p))

def grad_f(p):
    return -np.sum(((a.T*B)/(a@p)).T, axis=0)

def duality_gap(p): # duality gap
    ap = a@p
    # u = B/ap by corr. primal variables by KKT stationarity condition
    return -np.sum(B * np.log(B / ap)) - np.sum(B * np.log(ap))

print("======== solve the dual in p using CVXPY + Mosek ========")
begin = time() # time it
p_cp = cp.Variable(shape=(m,), nonneg=True)
obj_expr = 0
for i in range(n):
    obj_expr -= B[i] * cp.log(a[i] @ p_cp)
objective = cp.Minimize(obj_expr)
constraints = [cp.sum(p_cp) == sum_B] # linear constraints
prob = cp.Problem(objective, constraints) # define the optimization problem
cvxpy_opt_obj = prob.solve(solver="MOSEK", parallel=True) # solve it using SCS
p_cp = p_cp.value
print("cvxpy opt_obj = {}, time elapsed = {}".format(f(p_cp), time()-begin))

# try prox. grad. 
step = 3
max_iter = 80
p_pg = np.ones((m,))/sum_B
for iter_idx in range(1, max_iter+1):
    p_pg = proj_simplex(p_pg - step * grad_f(p_pg), s=sum_B)
    aTp = a@p_pg
    u = B/aTp
    print(np.max((a.T@u)))
    # print(iter_idx, np.linalg.norm(p_pg - p_cp))