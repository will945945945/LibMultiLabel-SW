#!/bin/bash

# basic seting
data_root="./datasets/binary_datasets"
log_root="./runs"

# Set up train command
task(){
train_cmd="python main.py"
train_cmd="${train_cmd} --linear"
train_cmd="${train_cmd} --linear_technique 1vsrest"
train_cmd="${train_cmd} --data_format svm"
train_cmd="${train_cmd} --monitor_metrics P@1"

for mode in trva trvate
do
    for dset in real-sim rcv1 ijcnn1 webspam
    #for dset in a9a #real-sim rcv1 ijcnn1 webspam
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
                s=3
                c=1
            elif [ "$mname" == "l2svm" ]; then
                # L2-regularized L2-loss support vector classification (dual)
                s=1
                c=1
            else
                # L2-regularized logistic regression (dual)
                s=0
                c=10
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

