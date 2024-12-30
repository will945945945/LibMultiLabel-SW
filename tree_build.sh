K=100
for seed in 1238 ; #1235 1236 1237 
do
    echo seed ${seed}
    # for data in eurlex4k wiki31k amazoncat13k 
    # do 
    #     echo ${data} l1 c 1 #-c 1
    #     python3 tree_build.py \
    #         --datapath datasets/${data}.pkl \
    #         --modelpath models/tree_K${K}_${data}_${seed}_l1_c1.pkl\
    #         --liblinear_options "-s 3 -c 1 -B 1 -q"\
    #         --treepath "treestructures/tree_structure_K${K}_${data}_${seed}.pkl"\
    #         --K ${K} \
    #         --seed ${seed} \
    #         --buildtree 1
            
    #     echo ${data} l2 c 1
    #     python3 tree_build.py \
    #         --datapath datasets/${data}.pkl \
    #         --modelpath models/tree_K${K}_${data}_${seed}_l2_c1.pkl\
    #         --liblinear_options "-s 1 -c 1 -B 1 -q"\
    #         --treepath "treestructures/tree_structure_K${K}_${data}_${seed}.pkl"\
    #         --K ${K} \
    #         --seed ${seed} \
    #     #    --buildtree 1

    #     echo ${data} lr c 10
    #     python3 tree_build.py \
    #         --datapath datasets/${data}.pkl \
    #         --modelpath models/tree_K${K}_${data}_${seed}_lr_c10.pkl\
    #         --liblinear_options "-s 0 -c 10 -B 1 -q"\
    #         --treepath "treestructures/tree_structure_K${K}_${data}_${seed}.pkl"\
    #         --K ${K} \
    #         --seed ${seed} \
    #     #    --buildtree 1
    # done

    for data in amazon670k
    do 
        # echo ${data} l1 c 1 #-c 1
        # python3 tree_build.py \
        #     --datapath datasets/${data}.pkl \
        #     --modelpath models/tree_K${K}_${data}_${seed}_l1_c1.pkl\
        #     --liblinear_options "-s 3 -c 1 -B 1 -q"\
        #     --treepath "treestructures/tree_structure_K${K}_${data}_${seed}.pkl"\
        #     --K ${K} \
        #     --seed ${seed} \
        #     # --buildtree 1
            
        echo ${data} l2 c 1
        python3 tree_build.py \
            --datapath datasets/${data}.pkl \
            --modelpath models/tree_K${K}_${data}_${seed}_l2_c1.pkl\
            --liblinear_options "-s 1 -c 1 -B 1 -q"\
            --treepath "treestructures/tree_structure_K${K}_${data}_${seed}.pkl"\
            --K ${K} \
            --seed ${seed} \
        #    --buildtree 1

        # echo ${data} lr c 10
        # python3 tree_build.py \
        #     --datapath datasets/${data}.pkl \
        #     --modelpath models/tree_K${K}_1_${data}_lr_c10.pkl\
        #     --liblinear_options "-s 0 -c 10 -B 1 -q"\
        #     --treepath "treestructures/tree_structure_K${K}_${data}.pkl"\
        # #    --buildtree 1
    done
done
