mkdir -p logs-tr
mkdir -p models

#for data in "amazon-670k-pecos";
for data in  "eur-lex" "wiki10-31k" "amazoncat-13k";
#for data in "amazon-670k-pecos";
#for data in "wiki-500k-pecos" "amazon-3m-pecos";
#for data in "wiki-500k-pecos";
do
  for num_models in 10;
  do
    for seed in 1 2 3 4 5 6 7 27 100 9527;
    #for seed in 100;
    do
      for idx in {0..9};
      do
        for K in 10;
        do
          #for sample_rate in 0.1 1.1;
          for sample_rate in 0.1;
          do
            #head="Rand-label-Forest-${num_models}"
            #head="Rand-label-partitions-No-replacement-${num_models}"
            #head="Rand-selection-${num_models}"
            head="Rand-label-Forest-No-replacement-${num_models}"
            param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
            name="${head}_${data}_${param}" 

            python3 train_random_label_forests_with_partitions.py \
	      --num_models ${num_models} --seed ${seed} --K ${K}  \
	      --sample_rate ${sample_rate} --idx ${idx} --datapath "./datasets/pickle-format/${data}" \
    	      |tee ./logs-tr/${name}.log
            #python3 train_random_selections.py \
            # if [ ! -f ./logs/${name}.log ];
            # then
	    #   echo "run ${name} !!"
            #   #python3 bagging_linear.py \
            #   python3 bagging_linear_training.py \
    	    #     --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
    	    #     --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
    	    #     |tee ./logs/${name}.log
            # fi
          done
        done
      done
    done
  done
done

#for data in  "amazon-3m-pecos";
# for data in  "wiki-500k-pecos";
# do
#   for num_models in 1;
#   do
#     for beam_width in 10;
#     do
#       for seed in 1;
#       do
#         for K in 100;
#         do
#           for sample_rate in 1.1;
#           do
#             head="Rand-label-Forest-${num_models}"
#             param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
#             name="${head}_${data}_${param}" 
#             
#             python3 bagging_linear_training.py \
#     	      --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
#     	      --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
#     	      |tee ./logs-tr/${name}.log
#             # if [ ! -f ./logs/${name}.log ];
#             # then
# 	    #   echo "run ${name} !!"
#             #   #python3 bagging_linear.py \
#             #   python3 bagging_linear_training.py \
#     	    #     --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
#     	    #     --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
#     	    #     |tee ./logs/${name}.log
#             # fi
#           done
#         done
#       done
#     done
#   done
# done

