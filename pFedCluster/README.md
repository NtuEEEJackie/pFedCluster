
**Note:** This repo is still in progress.

## Dependencies
* PyTorch = 1.12.0

## Quick Start

- Run FedAvg algorithm on CIFAR-10, skew partition, 10 clients, local iteration number is 200:

```console
python fedavg.py --gpu "7" --dataset 'cifar10' --partition 'noniid-skew' --n_parties 10 --num_local_iterations 200
```

- For fair comparisons on CIFAR-10, you can run the bash file. 

```console
sh run_cifar10.sh
```
- pfedCluster_cosine.py runs the experiment that we regularize the local model with the normalized aggregated model.



```