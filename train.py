"""Offline trainer — train the lung-cancer CNN on a REAL public dataset.

Downloads a real, labeled image dataset from the HuggingFace Hub (no auth /
API key needed), collapses its fine-grained labels to binary normal-vs-cancer,
trains the CNN with the project's pipeline, and saves an artifact the Streamlit
app loads at startup.

Why a separate script instead of training in the app: Render is CPU-only with
an ephemeral disk, so live training is slow and the model wouldn't survive a
restart. Run this once, commit the produced artifact, and the app serves it.

NOTE: TensorFlow 2.15 needs Python <= 3.11. Run in such an env:

    pip install -r requirements_train.txt
    python train.py                          # default: dorsar/lung-cancer (CT)
    python train.py --dataset keremberke/chest-xray-classification --config full
    python train.py --architecture resnet50 --epochs 20 --fine-tune-epochs 8

Outputs (committed so the app can load them):
    models/lung_cancer_model.keras
    models/labels.json
"""
import argparse
import json
import os

import numpy as np
import cv2
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from models.cnn_model import LungCancerCNN
from utils.data_preprocessing import MedicalImagePreprocessor
from utils.evaluation import MedicalModelEvaluator

IMG_SIZE = (224, 224)
CLASS_NAMES = ["normal", "cancer"]


def to_binary(label_name: str) -> int:
    """Any class whose name contains 'normal' is the negative class (0);
    every other label (adenocarcinoma, squamous/large-cell carcinoma,
    pneumonia, ...) is the positive class (1 = cancer/abnormal)."""
    return 0 if "normal" in label_name.lower() else 1


def _label_column(features) -> str:
    for col in ("label", "labels"):
        if col in features:
            return col
    raise ValueError(f"No label column found. Columns: {list(features)}")


def split_to_arrays(split, label_col, label_names, limit=0):
    """Turn a HuggingFace image split into (X[0..1], y) numpy arrays."""
    X, y = [], []
    for i, row in enumerate(split):
        if limit and i >= limit:
            break
        img = row["image"].convert("RGB")
        arr = cv2.resize(np.array(img), IMG_SIZE).astype("float32") / 255.0
        X.append(arr)
        y.append(to_binary(label_names[row[label_col]]))
    return np.array(X), np.array(y, dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dorsar/lung-cancer",
                    help="HuggingFace dataset id (image classification)")
    ap.add_argument("--config", default=None, help="Dataset config name (if any)")
    ap.add_argument("--architecture", default="efficientnetb0",
                    choices=["resnet50", "densenet121", "efficientnetb0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap examples per split (0 = all) for a quick smoke run")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    print(f"Loading dataset '{args.dataset}'"
          + (f" (config={args.config})" if args.config else "") + " ...")
    dsd = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)

    train_split_name = "train" if "train" in dsd else list(dsd.keys())[0]
    features = dsd[train_split_name].features
    label_col = _label_column(features)
    label_names = features[label_col].names
    print(f"Source classes ({len(label_names)}): {label_names}")
    print(f"Binary mapping: {[f'{n}->{to_binary(n)}' for n in label_names]}")

    X_train, y_train = split_to_arrays(dsd[train_split_name], label_col, label_names, args.limit)

    # Validation split (use the dataset's own, else carve one out of train)
    if "validation" in dsd or "valid" in dsd:
        val_name = "validation" if "validation" in dsd else "valid"
        X_val, y_val = split_to_arrays(dsd[val_name], label_col, label_names, args.limit)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  "
          f"(train normal/cancer = {int((y_train == 0).sum())}/{int((y_train == 1).sum())})")

    pre = MedicalImagePreprocessor(target_size=IMG_SIZE)
    class_weights = pre.calculate_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    train_gen, val_gen = pre.create_data_generators(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size, augment=True
    )

    model = LungCancerCNN(input_shape=(*IMG_SIZE, 3), num_classes=2,
                          architecture=args.architecture)
    model.build_model()
    model.compile_model()

    print(f"Training {args.architecture} for {args.epochs} epochs "
          f"(+{args.fine_tune_epochs} fine-tune) ...")
    model.train(
        train_gen, val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=1e-5,
    )

    # Honest evaluation on a held-out test split when available
    if "test" in dsd:
        X_test, y_test = split_to_arrays(dsd["test"], label_col, label_names, args.limit)
        results = MedicalModelEvaluator(CLASS_NAMES).evaluate_model(model.model, X_test, y_test)
        print("\n=== Test set performance ===")
        for k in ("accuracy", "sensitivity", "specificity", "auc", "f1_score"):
            v = results.get(k)
            print(f"  {k:12s}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k:12s}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "lung_cancer_model.keras")
    model.model.save(model_path)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "architecture": args.architecture,
            "input_shape": [*IMG_SIZE, 3],
            "source_dataset": args.dataset,
        }, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved labels -> {os.path.join(args.out_dir, 'labels.json')}")
    print("Commit both files; the app loads them automatically at startup.")


if __name__ == "__main__":
    main()
