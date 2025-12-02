

# GatorSC: Multi-Scale Cell and Gene Graphs with Mixture-of-Experts Fusion for Single-Cell Transcriptomics


## Requirements
- python : 3.9.12
- scanpy : 1.10.3
- sklearn : 1.1.1
- scipy : 1.9.0
- torch : 1.11.2
- torch-geometric : 2.1.0
- numpy : 1.24.4
- pandas : 2.2.3


## Project Structure

```bash
.
├── main.py            # Main training and evaluation loop
├── GatorSC.py           # Model architecture and loss functions
├── util.py            # Utility functions (seed setup, metrics)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
└── result.json        # Evaluation results output
```

## Usage

### **1. Prepare your input data**

Place your **.h5ad** spatial transcriptomics datasets in the `./data/` directory.

Each `.h5ad` file should contain:

* **`adata.X`** – Gene expression matrix 
* **`adata.obs`** – Cell metadata.
  The loader automatically searches typical label fields:

  ```
  ['cell_type']
  ```

Example dataset folder:

```bash
data/
 ├── BCs-PCs.h5ad
 ├── Chen.h5ad
 └── Emont.h5ad
```

---

### **2. Run training and evaluation**

Execute the main script to train and evaluate across datasets:

```bash
python main.py
```

#### Optional configuration inside `main.py`:

* `epochs`: number of training epochs per run 
* `batch_size`: number of samples per batch
* `lr`: learning rate
* `task`: application task ("clustering"/"imputation"/"annotation")

You can modify these using `--epochs --batch_size --lr --task`

---

### **3. Output files**

After training completes:

* Trained model checkpoints: `saved_models/`

  ```
  saved_models/
   ├── BCs-PCs_model_1_dict
   ├── BCs-PCs_model_imputation_1_dict
   ├── BCs-PCs_model_annotation_1_dict
   ...
  ```
* Evaluation results (accuracy, clustering, etc.): `result.json`

  ```json
  {
      "BCs-PCs": {'acc': , 'nmi': , 'ari': , 'homo': }
  }
  ```

---

## Datasets


##  Cross-Validation Strategy

To ensure robustness and generalizability across different tissue sections and data distributions, GatorST employs a **multi-run cross-validation approach**.

Specifically:

* For each dataset, the data loader (`loader_construction`) partitions samples into **train**, **validation**, and **test** sets using fixed proportions (80% / 10% / 10%).
* This process is **repeated across 10 random seeds** (`for run in range(10)`) to assess consistency.
* Each run reinitializes the model and randomizes data splits, ensuring that results reported in `result.json` reflect **average and variance** across independent splits.

This strategy provides a fair approximation of cross-validation while maintaining efficiency for large-scale ST datasets.

