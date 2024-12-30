# LibMultiLabel — a Library for Multi-class and Multi-label Classification

LibMultiLabel is a library for binary, multi-class, and multi-label classification. It has the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments
- Python: 3.8+
- CUDA: 11.8, 12.1 (if training neural networks by GPU)
- Pytorch: 2.0.1+

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS at their [website](https://pytorch.org/).

## Documentation
See the documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel

## For Probability Estimation on Binary Classification Datasets

### Install/Remove Environment

- For installing
```
make all
```

- For removing
```
make clean
```

### Prepare data

```
cd datasets
./prepare_data.sh
```

### Train models

```
./train_models.sh
```

### Conduct Experiments

Leave to do.

