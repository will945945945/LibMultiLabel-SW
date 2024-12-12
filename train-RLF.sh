mkdir -p logs-tr
mkdir -p models

num_models=100
sample_rate=0.1
K=100
for data in  "eur-lex";
do
  #for seed in 1 2 3 4 5 6 7 27 100 9527;
  for seed in 1;
  do
    for idx in {0..99};
    do
      head="Rand-label-Forest-${num_models}"
      param="seed=${seed}_K=${K}_sample-rate=${sample_rate}"
      name="${head}_${data}_${param}" 

      python3 train_random_label_forests.py \
        --num_models ${num_models} --seed ${seed} --K ${K} \
        --sample_rate ${sample_rate} --idx ${idx} --datapath "./datasets/pickle-format/${data}" \
          |tee ./logs-tr/${name}.log
    done
  done
done
