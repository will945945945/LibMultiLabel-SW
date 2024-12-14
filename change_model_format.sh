
mkdir -p csr_models

input_dir=models
output_dir=csr_models

#f="Rand-label-partitions-No-replacement_wiki10-31k_seed=27_K=100_sample-rate=0.1.model-0"
for f in ${input_dir}/*amazon-670k*;
do
  echo $(basename ${f})
  python3 change_sparse_format.py --input_dir ${input_dir} --modelname $(basename ${f}) --output_dir ${output_dir}
done
