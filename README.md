# Lung Cancer Image CNN Training

This project provides a PyTorch training and evaluation pipeline for lung cancer image classification using a CNN backbone (ResNet18).

## Setup for Testers

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ML_Project
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. **Create `data/` folder and upload dataset**

```bash
mkdir data
# Now upload the 10GB lung cancer dataset to the data/ folder
```

### Expected dataset structure

```text
data/
  train/
    cancer/
    non_cancer/
  test/
    cancer/
    non_cancer/
```

**OR** a single folder with class subfolders (script auto-splits into train/val/test):

```text
data/
  cancer/
  non_cancer/
```

## Training

### Option A: Command-line (VS Code Terminal)

```bash
python train.py --data-dir data --output-dir outputs --epochs 15 --batch-size 32 --pretrained
```

### Option B: Jupyter Notebook (VS Code or JupyterLab)

```bash
jupyter notebook lung_cancer_pipeline.ipynb
```

## Testing

### Option A: Command-line

```bash
python test.py --data-dir data --checkpoint outputs/best_model.pt
```

### Option B: Jupyter Notebook

Run evaluation cells in `lung_cancer_pipeline.ipynb`

## Results

- Best model checkpoint: `outputs/best_model.pt`
- Evaluation report: `outputs/evaluation_report.json`
- Training plots: `outputs/training_history.png`
- Confusion matrix: `outputs/confusion_matrix.png`

## Notes

- **Generic for any image classification task**: This code works with any image dataset (medical imaging, plant disease, etc.), not just lung cancer. Just organize your data into class folders.
- **CT scan support**: If your images are CT scans or grayscale medical images, the pipeline converts them to 3 channels automatically so the pretrained backbones can use them.
- The code uses image augmentation for training and normalization for evaluation.
- It is configured to use CUDA automatically when available.
- For a 10 GB dataset, increase `--num-workers` if your machine has enough CPU cores and memory.
- Do NOT commit `data/`, `outputs/`, or `.venv/` to GitHub—they're in `.gitignore`
