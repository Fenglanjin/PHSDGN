# PHSDGN

This repository provides the PyTorch implementation of **PHSDGN**: a
physics-embedded high-order spatiotemporal dynamic graph network for power
system inertia estimation using post-disturbance PMU measurements.

The model combines topology-masked dynamic graph learning, graph-Laplacian
high-order propagation, temporal differencing, and a dual-track temporal readout
to estimate the equivalent system inertia of the IEEE 24-bus system.

## Repository Structure

```text
.
|-- dataset.py         # IEEE 24-bus dataset loading, PMU mapping, data splits
|-- model.py           # PHSDGN model definition
|-- utils.py           # Dynamic graph, physics-embedded propagation, readout modules
|-- train.py           # Training entry point
|-- eval.py            # Checkpoint evaluation entry point
|-- metrics.py         # MSE, MAPE, ACC, and R2 metrics
|-- requirements.txt   # Python dependencies
`-- README.md
```

Large datasets, checkpoints, logs, paper files, and generated figures are not
intended to be tracked in Git.

## Dataset

The two datasets used in this project can be obtained from Figshare:

<https://figshare.com/articles/dataset/Raw_Dataset_for_IEEE_24-Bus_System_Freuqency_Dynamics_Measurements/23576769>

After downloading, place the CSV files in the repository root or pass their
paths with `--data_path`. The expected filenames used by the default scripts are:

```text
train_3rd_wrv.csv
train_3rd_wrv_SNR45db.csv
```

These files are intentionally excluded from version control because of their
size.

## Environment

The code has been used with Python 3.10+ and PyTorch. A minimal environment can
be created as follows:

```bash
conda create -n phsdgn python=3.10
conda activate phsdgn
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if GPU acceleration is
needed. See the official PyTorch installation instructions for platform-specific
commands.

## Training

Train PHSDGN on the 45 dB noisy dataset:

```bash
python train.py \
  --data_path train_3rd_wrv_SNR45db.csv \
  --device cuda:0 \
  --batch_size 8 \
  --K 3 \
  --Kt 7 \
  --alpha 0.1 \
  --use_dynamic_graph true \
  --use_physics_conv true \
  --use_std_pooling true
```

Train on the clean dataset:

```bash
python train.py --data_path train_3rd_wrv.csv --device cuda:0
```

The script saves checkpoints under `output/` and logs under `logs/`. Checkpoints
store the model weights, training arguments, normalization statistics, and split
metadata so that validation or test evaluation can be reproduced.

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py \
  --data_path train_3rd_wrv_SNR45db.csv \
  --ckpt_path output/PHSDGN_train_3rd_wrv_SNR45db_DG1_PE1_SP1_best.pt \
  --eval_split val \
  --device cuda:0 \
  --batch_size 8
```

If `--ckpt_path` is omitted, `eval.py` builds the default checkpoint name from
the command-line configuration.

## Metrics

The repository reports:

- `MSE`: mean squared error.
- `MAPE`: mean absolute percentage error.
- `ACC`: percentage of predictions whose relative error is within the specified
  tolerance. The default tolerance is `5%` (`--acc_tol 0.05`).
- `R2`: coefficient of determination.

## Notes for Reproducibility

- The default data split is `80%` training and `20%` validation with no test
  samples. This follows the current experimental scripts.
- Set `--split_shuffle true` to shuffle before splitting. The split seed is
  controlled by `--split_seed`.
- Because the raw datasets are hosted externally, this repository does not
  include the CSV files or trained checkpoints.

## Citation

If you use this code, please cite the associated paper when it becomes
available.
