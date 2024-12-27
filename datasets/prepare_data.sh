#!/bin/bash

root="binary_datasets"
mkdir -p $root
cd $root

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

## Dispatch, soft-link and split
for i in a9a ijcnn1 rcv1 real-sim webspam
do
    ds=dataset_$i
    mkdir -p $ds
    mv $i* $ds/
    cd $ds
    if [ "$i" == "a9a" ]; then
        ln -sf $i   trva.svm
        ln -sf $i.t te.svm
    elif [ "$i" == "ijcnn1" ]; then
        ln -sf $i   trva.svm
        ln -sf $i.t te.svm
    elif [ "$i" == "rcv1" ]; then
        ln -sf ${i}_train.binary trva.svm
        ln -sf ${i}_test.binary  te.svm
    elif [ "$i" == "real-sim" ]; then
        ../../random_split.sh $i 90
        ln -sf ${i}.trva trva.svm
        ln -sf ${i}.te   te.svm
    else
        ../../random_split.sh ${i}_wc_normalized_unigram.svm 90
        ln -sf ${i}_wc_normalized_unigram.svm.trva trva.svm
        ln -sf ${i}_wc_normalized_unigram.svm.te   te.svm
    fi
    cd ../
done

echo "All done!"

