#!/bin/bash
set -e
# basic seting
data_root="./datasets/binary_datasets"
log_root="./runs"

# L1 search func
#!/bin/bash

# Function to find the best C value
find_best_c() {
    local train_path=$1
    local best_c=0
    local best_acc=0

    for c in 0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024; do

        acc=$(./liblinear/train -s 3 -c $c -v 5 $train_path | grep "Cross Validation Accuracy" | awk '{print $5}' | tr -d '%')

        if (( $(echo "$acc > $best_acc" | bc -l) )); then
            best_acc=$acc
            best_c=$c
        fi
    done
    echo "$best_c"
}


# Set up train command
task(){
train_cmd="python main.py"
train_cmd="${train_cmd} --linear"
train_cmd="${train_cmd} --linear_technique 1vsrest"
train_cmd="${train_cmd} --data_format svm"
train_cmd="${train_cmd} --monitor_metrics P@1"

# for dset in a9a real-sim rcv1 ijcnn1 webspam
    # do
for mode in trva trvate
do
    for dset in rcv1_reverse
    do
        for mname in l1svm l2svm lr
        do
            data_path="$data_root/dataset_$dset"
            cmd="${train_cmd} --data_name $dset"
            cmd="${cmd} --result_dir $log_root/${mode}/"
            cmd="${cmd} --training_file $data_path/${mode}.svm"
            cmd="${cmd} --test_file $data_path/te.svm"
            if [ "$mname" == "l1svm" ]; then
                # L2-regularized L1-loss support vector classification (dual)
                echo "Begin Search"
                c=$(find_best_c $data_path/${mode}.svm)
                s=3
            elif [ "$mname" == "l2svm" ]; then
                # L2-regularized L2-loss support vector classification (dual)
                s=2
                c=$(./liblinear/train -s 2  -v 5 -C $data_path/${mode}.svm | grep "Best C" | awk '{print $4}')
            else
                # L2-regularized logistic regression (dual)
                s=0
                c=$(./liblinear/train -s 0 -v 5 -C $data_path/${mode}.svm | grep "Best C" | awk '{print $4}')
                echo "$c"
            fi
            cmd="${cmd} --liblinear_options='-s $s -c $c'"
            cmd="${cmd} --model_name ${mname}_c${c}"
            echo "${cmd}"
        done
    done
done
}

# Check
task
wait

# Run
task | xargs -0 -d '\n' -P 3 -I {} sh -c {}

