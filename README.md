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
├── GatorSC.py         # Model architecture and loss functions
├── util.py            # Utility functions (seed setup, metrics)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
└── saved_results/     # Evaluation results output
```

## Usage

### **1. Prepare your input data**

Place your **.h5ad** datasets in the `./data/` directory.

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
 ...
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
* `task`: application task (clustering/imputation/annotation)

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
* Evaluation results (clustering/imputation/annotation): `saved_results/`

  ```
  saved_models/
   ├── BCs-PCs_clustering_result.json
   ├── BCs-PCs_imputation_result.json
   ├── BCs-PCs_annotation_result.json
   ...
  ```

---

## Datasets
- BCs-PCs : https://treg-gut-niches.cellgeni.sanger.ac.uk
- Chen : https://cellxgene.cziscience.com/collections/10ec9198-584e-4a7e-8a24-4a332915a4ef
- Emont : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176171
- Global : https://treg-gut-niches.cellgeni.sanger.ac.uk
- HTAPP-213-SMP-6752 : https://cellxgene.cziscience.com/collections/a96133de-e951-4e2d-ace6-59db8b3bfb1d
- HTAPP-313-SMP-932 : https://cellxgene.cziscience.com/collections/a96133de-e951-4e2d-ace6-59db8b3bfb1d
- HTAPP-783-SMP-4081 : https://cellxgene.cziscience.com/collections/a96133de-e951-4e2d-ace6-59db8b3bfb1d
- Quake-Diaphragm : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Quake-Limb Muscle : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Quake-Lung : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Quake-Trachea : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Zeisel : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361
- Mouse ES : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525
- Pancreas : https://ndownloader.figshare.com/files/36086813
- AMB3 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746
- AMB16 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746
- AMB92 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746
- TM : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Zheng : https://figshare.com/articles/dataset/pbmc68k_h5ad/14465568?file=27686886
