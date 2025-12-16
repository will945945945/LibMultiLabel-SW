set -e

for seed in 1 2 3 4 5;
do
    python main_comb.py --search tuned --seed $seed --shuffle-folds
    python main_comb.py --search untuned --seed $seed --shuffle-folds
    python main.py --search tuned --seed $seed --shuffle-folds
    python main.py --search untuned --seed $seed --shuffle-folds
done
