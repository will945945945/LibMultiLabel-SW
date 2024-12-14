mkdir -p logs-tr
mkdir -p models

num_models=10
K=10
#for data in  "amazoncat-13k";
for data in "amazon-670k-pecos";
do
  for seed in 1 2 3 4 5 6 7 27 100 9527;
  #for seed in 100;
  do
    for idx in {0..9};
    do
      head="Rand-label-Forest-No-replacement-${num_models}"
      param="seed=${seed}_K=${K}"
      name="${head}_${data}_${param}" 

      python3 train_random_label_forests_with_partitions.py \
        --num_models ${num_models} --seed ${seed} --K ${K}  \
        --idx ${idx} --datapath "./datasets/pickle-format/${data}" \
          |tee ./logs-tr/${name}.log
    done
  done
done
