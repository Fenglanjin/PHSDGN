# PHSDGN

This project provides the PyTorch implementation of **PHSDGN**.

## Project Structure

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

## Dataset

The two datasets used in this project can be obtained from Figshare:

<https://figshare.com/articles/dataset/Raw_Dataset_for_IEEE_24-Bus_System_Freuqency_Dynamics_Measurements/23576769>

After downloading, place the CSV files in the repository root or pass their
paths with `--data_path`. The expected filenames used by the default scripts are:

```text
train_3rd_wrv.csv
train_3rd_wrv_SNR45db.csv
```

## Environment

The code has been used with Python 3.10+ and PyTorch. A minimal environment can
be created as follows:

```bash
conda create -n phsdgn python=3.10
conda activate phsdgn
pip install -r requirements.txt
```

## Training

Train on the clean dataset:

```bash
python train.py --data_path train_3rd_wrv.csv --device cuda:0
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py \
  --data_path train_3rd_wrv.csv \
  --ckpt_path output/PHSDGN_train_3rd_wrv_SNR45db_DG1_PE1_SP1_best.pt \
  --eval_split val \
  --device cuda:0 \
  --batch_size 8
```

## Citation

Please cite the associated paper when it becomes available if you use this code.
