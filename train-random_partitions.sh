mkdir -p logs-tr
mkdir -p models

for data in  "eur-lex" "wiki10-31k" "amazoncat-13k" "amazon-670k-pecos";
do
  #for seed in 1 2 3 4 5 6 7 27 100 9527;
  for seed in 1;
  do
    for K in 100;
    do
      num_models=1
      idx=0
      head="Rand-label-partitions-No-replacement-${num_models}"
      param="seed=${seed}_K=${K}"
      name="${head}_${data}_${param}" 

      python3 train_random_partitions.py \
        --num_models ${num_models} --seed ${seed} --K ${K} \
        --idx ${idx} --datapath "./datasets/pickle-format/${data}" \
        |tee ./logs-tr/${name}.log
    done
    for K in 10;
    do
      num_models=10
      for idx in {0..9};
      do
        head="Rand-label-partitions-No-replacement-${num_models}"
        param="seed=${seed}_K=${K}"
        name="${head}_${data}_${param}" 

        python3 train_random_partitions.py \
          --num_models ${num_models} --seed ${seed} --K ${K} \
          --idx ${idx} --datapath "./datasets/pickle-format/${data}" \
          |tee ./logs-tr/${name}.log
      done
    done
  done
done
