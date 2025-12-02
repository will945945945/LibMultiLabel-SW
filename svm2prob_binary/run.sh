set -e

python main.py tuned
python main.py untuned
python main_comb.py untuned
python main_comb.py tuned