# run across different seeds and different sizes

# for sd = 1, ..., 10
# (n, m) = (25, 50), (50, 100), (100, 150), (150, 200)
# all 4 distributions
# 

run_experiments_for_one_seed() {
    for n in 200 400 600 800
    do 
        let m=$n*2
        python experiment_ql.py --sd=$1 --n=$n --m=$m --distr=unif --da=$2 --fm=w
        python experiment_ql.py --sd=$1 --n=$n --m=$m --distr=normal --da=$2 --fm=a
        python experiment_ql.py --sd=$1 --n=$n --m=$m --distr=exp --da=$2 --fm=a
        python experiment_ql.py --sd=$1 --n=$n --m=$m --distr=log_normal --da=$2 --fm=a
    done
}

# run_experiments_for_one_seed 15

for sd in {1..10}
do 
    run_experiments_for_one_seed $sd 5e-6 &
done
wait

for sd in {11..20}
do 
    run_experiments_for_one_seed $sd 5e-6 &
done
wait

for sd in {21..30}
do 
    run_experiments_for_one_seed $sd 5e-6 &
done
wait

echo 'now plot...'
python plot_ql.py
echo 'all set.'
