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
# additional
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a0a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a0a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/liver-disorders
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/liver-disorders.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon.t
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/sonar_scale
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms

## Decompress
echo "Decompressing..."
for i in `ls *.bz2`; do bzip2 -d $i; done
xz --decompress webspam_wc_normalized_unigram.svm.xz
echo "Completed"
cd ../

## Dispatch, soft-link and split
# Datasets you want processed
target_datasets="a0a a1a a2a a3a a4a a5a a6a a7a a8a \
                 breast-cancer_scale ionosphere_scale diabetes_scale \
                 liver-disorders \
                 madelon \
                 sonar_scale \
                 skin_nonskin phishing mushrooms \
                 a9a ijcnn1 rcv1 real-sim webspam"

for i in $target_datasets; do
    ds=dataset_$i
    mkdir -p $ds
    cd $ds

    echo "Processing dataset: $i ..."

    case "$i" in

        # ------------------------------------------------
        # Existing datasets
        # ------------------------------------------------

        a0a|a1a|a2a|a3a|a4a|a5a|a6a|a7a|a8a|a9a|ijcnn1|"liver-disorders"|madelon|gisette_scale)
            ln -sf ../raw_data/$i   trva.svm
            ln -sf ../raw_data/$i.t te.svm
            cat trva.svm te.svm > trvate.svm
            ;;

        "rcv1")
            ln -sf ../raw_data/rcv1_train.binary trva.svm
            ln -sf ../raw_data/rcv1_test.binary  te.svm
            cat trva.svm te.svm > trvate.svm
            ;;

        "real-sim")
            ../../random_split.sh ../raw_data/real-sim 90
            ln -sf ../raw_data/real-sim.trva trva.svm
            ln -sf ../raw_data/real-sim.te   te.svm
            ln -sf ../raw_data/real-sim      trvate.svm
            ;;

        breast-cancer_scale|ionosphere_scale|diabetes_scale|sonar_scale|skin_nonskin|phishing|mushrooms)
            ../../random_split.sh ../raw_data/$i 90
            ln -sf ../raw_data/$i.trva trva.svm
            ln -sf ../raw_data/$i.te   te.svm
            ln -sf ../raw_data/$i      trvate.svm
            ;;
        
        "webspam")
            ../../random_split.sh ../raw_data/webspam_wc_normalized_unigram.svm 90
            ln -sf ../raw_data/webspam_wc_normalized_unigram.svm.trva trva.svm
            ln -sf ../raw_data/webspam_wc_normalized_unigram.svm.te   te.svm
            ln -sf ../raw_data/webspam_wc_normalized_unigram.svm      trvate.svm
            ;;

        *)
            echo "Unknown dataset type: $i"
            ;;

    esac

    cd ../
done


echo "All done!"

