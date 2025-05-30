
K=100
width=10
slurm_model_path=/l/users/kuanting.chen/models
for num in 1234 #1235 1236 1237 1238 ; 
do
    for data in amazoncat13k #wiki31k #mimic amazoncat13k # amazoncat13k eurlex4k #amazon670k
    do 
        for c in 64; #2 4; #8 16 32 64 ; #8 16 32 64 ; #0.25 0.5 1 2 4; # 8 16 32 64 ; #0.125 0.0625; #
        do
            # echo ${data} l1 c=${c}
            # python3 ensemble_test_tree_cv.py \
            #     --datapath datasets/${data}.pkl \
            #     --dataname datasets/${data}_5folds \
            #     --modelname ${slurm_model_path}/tree_K${K}_${data}_${num}_l1_c${c} \
            #     --beamwidth ${width} \
            #     --modeltype l1

            # echo ${data} l2 c=${c}
            # python3 ensemble_test_tree_cv.py \
            #     --datapath datasets/${data}.pkl \
            #     --dataname datasets/${data}_5folds \
            #     --modelname ${slurm_model_path}/tree_K${K}_${data}_${num}_l2_c${c} \
            #     --beamwidth ${width} \
            #     --modeltype l2

            echo ${data} lr c=${c}
            python3 ensemble_test_tree_cv.py \
                --datapath datasets/${data}.pkl \
                --dataname datasets/${data}_5folds \
                --modelname ${slurm_model_path}/tree_K${K}_${data}_${num}_lr_c${c} \
                --beamwidth ${width} \
                --modeltype lr
        done
    done
done