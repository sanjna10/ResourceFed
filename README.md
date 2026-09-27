# FedLearnLab

A lightweight PyTorch implementation of **Federated Averaging (FedAvg)** for experimenting with federated learning using IID and non-IID data.

Based on *Communication-Efficient Learning of Deep Networks from Decentralized Data* (McMahan et al., 2017).

## Architecture

```text
                 Global Model
                      |
               Select Clients
                      |
          +-----------+-----------+
          |           |           |
       Client 1    Client 2    Client N
          |           |           |
       Local       Local       Local
       Training    Training    Training
          |           |           |
          +-----------+-----------+
                      |
                Aggregate Models
                      |
                      v
               Updated Global Model
```

Each selected client receives the current global model, trains it on its local data, and returns the updated model parameters. The client models are then aggregated to update the global model.

## Project Structure

```text
FedLearnLab/
├── data/                     # Datasets
├── save/                     # Experiment results
└── src/
    ├── train_federated.py    # Federated training
    ├── train_centralized.py  # Centralized baseline
    ├── client.py             # Local client training
    ├── data_partition.py     # IID / non-IID partitioning
    ├── aggregation.py        # Model aggregation
    ├── models.py             # MLP / CNN models
    └── config.py             # Experiment configuration
```

## Features

- Federated Averaging (FedAvg)
- IID and non-IID data partitioning
- Random client selection
- Local client training
- Centralized training baseline
- MNIST, Fashion-MNIST and CIFAR-10
- MLP and CNN models

## Key Parameters

| Parameter | Description |
|---|---|
| `num_users` | Total number of clients |
| `frac` | Fraction of clients selected per round |
| `local_ep` | Number of local epochs |
| `local_bs` | Local batch size |
| `iid` | IID or non-IID data distribution |


## Run

### Federated Learning — IID

```bash
python src/train_federated.py \
    --dataset mnist \
    --model mlp \
    --iid 1
```

### Federated Learning — Non-IID

```bash
python src/train_federated.py \
    --dataset mnist \
    --model mlp \
    --iid 0
```

### Centralized Baseline

```bash
python src/train_centralized.py \
    --dataset mnist \
    --model mlp
```

## Note

The clients in this project are **simulated on a single machine**. The implementation focuses on the FedAvg training process rather than communication between real mobile or edge devices.



