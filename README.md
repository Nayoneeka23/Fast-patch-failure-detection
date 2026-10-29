# Fast Failure Detection in LLM-generated Patches

Predict, **without running any tests**, whether a patch produced by an LLM agent will fail,
so clearly wrong patches can be filtered out before Continuous Integration.

- Dataset: [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) (split: `ticks`)
- Input: the agent trajectory (`messages`) and the final `patch`
- Target: `resolved` (True / False)
- Models: small, fast scikit-learn models (Logistic Regression, Random Forest, SVM)

## Setup (Windows, PowerShell)

```powershell
git clone https://github.com/Nayoneeka23/Fast-patch-failure-detection.git
cd Fast-patch-failure-detection

# Create the virtual environment OUTSIDE the repo (the repo may live in OneDrive)
python -m venv $HOME\venvs\fastpatch
& $HOME\venvs\fastpatch\Scripts\Activate.ps1

pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → `...\venvs\fastpatch\Scripts\python.exe`.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `src/load_data.py` | Download the `ticks` split, build a random ~500-row mini dataset *(to do)* |
| 2 | `src/explore.py` | Dataset statistics and data-quality checks *(to do)* |
| 3 | `src/preprocess.py` | Cleaning, parsing trajectories *(to do)* |
| 4 | `src/features.py` | Hand-made features: one row of numbers per patch *(to do)* |
| 5 | `src/train.py` | Train and compare LogReg / RF / SVM *(to do)* |
| 6 | `src/evaluate.py` | Run a saved model on any dataset in the same format and report metrics *(to do)* |

The pipeline is developed on the mini dataset first, then run unchanged on the full split.

## Repository layout

```
data/        datasets (not committed, created by scripts)
docs/        supervisor slides and reference material
notebooks/   quick exploration only
src/         the pipeline
results/     metrics tables and plots
```
