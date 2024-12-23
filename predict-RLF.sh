mkdir -p logs
mkdir -p models

beam_width=10;
sample_rate=0.1;
#for data in "amazon-670k-pecos";
for data in "eur-lex" "wiki10-31k";
do
  for seed in 1 2 3 4 5 6 7 27 100 9527;
  do
    num_models=100;
    K=100;

    head="Rand-label-Forest-${num_models}"
    param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
    name="${head}_${data}_${param}" 

    if [ ! -f ./logs/${name}.log ];
    then
      export MKL_INTERFACE_LAYER=ILP64
      python3 predict_random_label_forests.py \
        --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
        --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
        |tee ./logs/${name}.log
    fi
  done
done
