# Training the model on a real public dataset

The app can run a **real** trained model. Because Render is CPU-only with an
ephemeral disk, training happens **offline** with `train.py`; you commit the
resulting artifact and the app loads it automatically at startup.

## What the trainer does

`train.py`:
1. Downloads a real, **labeled** image dataset from the HuggingFace Hub
   (no account / API key required).
2. Collapses its fine-grained labels to binary **normal vs cancer**
   (`"normal"` in the class name → `0`, every other class → `1`).
3. Trains the CNN with architecture-correct preprocessing + fine-tuning.
4. Evaluates on the held-out test split (accuracy, sensitivity, specificity, AUC).
5. Saves `models/lung_cancer_model.keras` and `models/labels.json`.

Default dataset: [`dorsar/lung-cancer`](https://huggingface.co/datasets/dorsar/lung-cancer)
— real lung CT scans (adenocarcinoma / squamous / large-cell carcinoma / normal).

## Run it (once)

> TensorFlow 2.15 requires **Python ≤ 3.11**. Use a 3.10/3.11 venv or Google Colab.

```bash
pip install -r requirements_train.txt

# default: dorsar/lung-cancer (CT, normal vs cancer)
python train.py

# or a chest X-ray dataset (NORMAL vs PNEUMONIA -> normal vs abnormal)
python train.py --dataset keremberke/chest-xray-classification --config full

# quick smoke run on a few images
python train.py --limit 40 --epochs 1 --fine-tune-epochs 0
```

Useful flags: `--architecture {efficientnetb0,resnet50,densenet121}`,
`--epochs`, `--fine-tune-epochs`, `--batch-size`, `--limit`.

## Commit the artifact

```bash
git add models/lung_cancer_model.keras models/labels.json
git commit -m "Add trained lung-cancer model artifact"
git push
```

EfficientNetB0 produces a ~20 MB `.keras` file — fine for a normal git commit
(well under GitHub's 100 MB limit). For larger backbones, use Git LFS:
`git lfs track "models/*.keras"`.

Once pushed and deployed, the home page and Prediction page automatically detect
the artifact (via `utils/model_state.ensure_model_loaded`) and switch from
"image diagnostics only" to full AI classification with Grad-CAM.

## Honesty note

These public datasets are research-grade and the binary mapping is a
simplification. The result is a genuine, real-data classifier suitable for a
demo/portfolio — **not** a clinically validated diagnostic tool.
