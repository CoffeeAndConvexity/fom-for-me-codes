# two things: B[i] ~ |np.random.normal| + 0.5 v.s. B == 1

# run across different seeds and different sizes

# for sd = 1, ..., 10
# (n, m) = (25, 50), (50, 100), (100, 150), (150, 200)
# all 4 distributions
# 

run_experiments_for_one_seed() {
    for n in 100 200 300 400
    do 
        let m=$n*2
        python experiment_linear.py --sd=$1 --n=$n --m=$m --distr=unif --da=$2 --fm=w
        python experiment_linear.py --sd=$1 --n=$n --m=$m --distr=normal --da=$2 --fm=a
        python experiment_linear.py --sd=$1 --n=$n --m=$m --distr=exp --da=$2 --fm=a
        python experiment_linear.py --sd=$1 --n=$n --m=$m --distr=log_normal --da=$2 --fm=a
    done
    
    # python experiment_pgls_and_pr.py --sd=$1 --n=100 --m=200 --distr=unif --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=100 --m=200 --distr=normal --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=100 --m=200 --distr=exp --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=100 --m=200 --distr=log_normal --da=$2 --fm=a

    # python experiment_pgls_and_pr.py --sd=$1 --n=150 --m=300 --distr=unif --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=150 --m=300 --distr=normal --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=150 --m=300 --distr=exp --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=150 --m=300 --distr=log_normal --da=$2 --fm=a
    
    # python experiment_pgls_and_pr.py --sd=$1 --n=200 --m=400 --distr=unif --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=200 --m=400 --distr=normal --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=200 --m=400 --distr=exp --da=$2 --fm=a
    # python experiment_pgls_and_pr.py --sd=$1 --n=200 --m=400 --distr=log_normal --da=$2 --fm=a
}

# run_experiments_for_one_seed 15

for sd in {1..10}
do 
    run_experiments_for_one_seed $sd 1e-3 &
done

wait

for sd in {11..20}
do 
    run_experiments_for_one_seed $sd 1e-3 &
done

wait

for sd in {21..30}
do 
    run_experiments_for_one_seed $sd 1e-3 &
done

wait

python plot_linear.py
echo "all set."