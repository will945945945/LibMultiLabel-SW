c=1
K=100
width=-1
for num in 1234 #1235 1236 1237 1238 ; 
do
echo num ${num}
    for data in eurlex4k wiki31k amazoncat13k #amazon670k
    do 
        echo ${data} l1 c=${c}
        python3 test_ovr.py \
            --datapath datasets/${data}.pkl \
            --modelpath models/ovr_${data}_${num}_l1_c${c}.pkl \
            --modeltype l1

        echo ${data} l2 c=${c}
        python3 test_ovr.py \
            --datapath datasets/${data}.pkl \
            --modelpath models/ovr_${data}_${num}_l2_c${c}.pkl \
            --modeltype l2

        echo ${data} lr c=10
        python3 test_ovr.py \
            --datapath datasets/${data}.pkl \
            --modelpath models/ovr_${data}_${num}_lr_c10.pkl \
            --modeltype lr

    done
done