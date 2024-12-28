#!/bin/bash

root="binary_datasets"
mkdir -p $root/raw_data
cd $root/raw_data

## Download
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2
for i in t tr val; do wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/ijcnn1.$i.bz2; done
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/a9a
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/a9a.t
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/real-sim.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.xz
#wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.xz

## Decompress
echo "Decompressing..."
for i in `ls *.bz2`; do bzip2 -d $i; done
xz --decompress webspam_wc_normalized_unigram.svm.xz
#xz --decompress webspam_wc_normalized_trigram.svm.xz
echo "Completed"
cd ../

## Dispatch, soft-link and split
for i in a9a ijcnn1 rcv1 real-sim webspam
do
    ds=dataset_$i
    mkdir -p $ds
    cd $ds
    if [ "$i" == "a9a" ]; then
        ln -sf ../raw_data/$i   trva.svm
        ln -sf ../raw_data/$i.t te.svm
        cat trva.svm te.svm > trvate.svm
    elif [ "$i" == "ijcnn1" ]; then
        ln -sf ../raw_data/$i   trva.svm
        ln -sf ../raw_data/$i.t te.svm
        cat trva.svm te.svm > trvate.svm
    elif [ "$i" == "rcv1" ]; then
        ln -sf ../raw_data/${i}_train.binary trva.svm
        ln -sf ../raw_data/${i}_test.binary  te.svm
        cat trva.svm te.svm > trvate.svm
    elif [ "$i" == "real-sim" ]; then
        ../../random_split.sh ../raw_data/$i 90
        ln -sf ../raw_data/${i}.trva trva.svm
        ln -sf ../raw_data/${i}.te   te.svm
        ln -sf ../raw_data/$i        trvate.svm
    else
        ../../random_split.sh ../raw_data/${i}_wc_normalized_unigram.svm 90
        ln -sf ../raw_data/${i}_wc_normalized_unigram.svm.trva trva.svm
        ln -sf ../raw_data/${i}_wc_normalized_unigram.svm.te   te.svm
        ln -sf ../raw_data/${i}_wc_normalized_unigram.svm      trvate.svm
    fi
    cd ../
done

echo "All done!"

