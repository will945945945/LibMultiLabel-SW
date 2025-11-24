set -e

python main.py untuned
python main.py tuned
python main_comb.py untuned
python main_comb.py tuned