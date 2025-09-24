#!/usr/bin/env python3
"""
Cancer-GAN (simple version)

Goal
- Make extra malignant rows so the data is more balanced.
- Use a small GAN to create them.
- Test if this helps a plain classifier.
- Keep things simple, clear, and short.
"""

import os
import time
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "breast-cancer.csv")  # expects your csv here
OUT_METRICS = os.path.join(OUT_DIR, "cancer_gan_metrics.csv")
OUT_AUG = os.path.join(OUT_DIR, "breast-cancer-gan-augmented.csv")
OUT_LOG = os.path.join(OUT_DIR, "cancer_gan.log")


def log(msg: str):
    """Write a short line to the log file and print it. Keep it clear."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# 1) Load data
def load_breast_cancer(path: str) -> pd.DataFrame:
    """Read the csv. Expect a 'diagnosis' column with 'B' and 'M'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # Drop id if it exists. It is not a feature.
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "diagnosis" not in df.columns:
        raise ValueError("Missing 'diagnosis' column.")
    return df


# 2) Train/test split
def make_splits(df: pd.DataFrame, test_size: float = 0.2):
    """Split rows. Keep the label ratio the same in both sets."""
    y = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    X = df.drop(columns=["diagnosis"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    return X_tr.reset_index(drop=True), X_te.reset_index(drop=True), y_tr.reset_index(drop=True), y_te.reset_index(drop=True)


# 3) Small Conditional GAN
class Generator(nn.Module):
    """
    Make fake feature rows from noise + label.
    We only use label=1 (malignant), but we keep the label input for clarity.
    """
    def __init__(self, noise_dim: int, label_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + label_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        x = torch.cat([z, y], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Tell if a row is real or fake, given the label context."""
    def __init__(self, in_dim: int, label_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + label_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        v = torch.cat([x, y], dim=1)
        return self.net(v)


@dataclass
class GANConfig:
    noise_dim: int = 32
    label_dim: int = 2   # 0 = benign, 1 = malignant
    hidden: int = 64
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    batch_size: int = 64
    epochs: int = 500
    device: str = "cpu"


def one_hot(labels: torch.Tensor, num_classes: int):
    """Basic one-hot. Simple and fine for our need."""
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def train_gan_on_malignant(X_train_scaled: np.ndarray, y_train: np.ndarray, cfg: GANConfig) -> Tuple[Generator, Discriminator]:
    """
    Train on malignant rows only.
    Idea: the generator learns the shape of malignant records.
    """
    device = torch.device(cfg.device)
    mal_idx = np.where(y_train == 1)[0]
    X_mal = X_train_scaled[mal_idx]
    y_mal = y_train[mal_idx]
    if X_mal.shape[0] < 10:
        raise ValueError("Too few malignant rows to train a GAN.")

    X_mal_t = torch.tensor(X_mal, dtype=torch.float32, device=device)
    y_mal_t = torch.tensor(y_mal, dtype=torch.long, device=device)

    gen = Generator(cfg.noise_dim, cfg.label_dim, out_dim=X_mal.shape[1], hidden=cfg.hidden).to(device)
    disc = Discriminator(in_dim=X_mal.shape[1], label_dim=cfg.label_dim, hidden=cfg.hidden).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=cfg.lr_g)
    opt_d = optim.Adam(disc.parameters(), lr=cfg.lr_d)
    bce = nn.BCELoss()

    n = X_mal_t.size(0)

    for epoch in range(cfg.epochs):
        # Shuffle rows each epoch
        perm = torch.randperm(n, device=device)
        d_loss_epoch, g_loss_epoch = 0.0, 0.0

        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i+cfg.batch_size]
            real_x = X_mal_t[idx]
            labels = y_mal_t[idx]  # should be all ones here
            y_onehot = one_hot(labels, cfg.label_dim)

            # Train Discriminator
            disc.train(); gen.train()
            opt_d.zero_grad()

            # Real
            real_prob = disc(real_x, y_onehot)
            loss_real = bce(real_prob, torch.ones_like(real_prob))

            # Fake
            z = torch.randn(real_x.size(0), cfg.noise_dim, device=device)
            cond = one_hot(torch.ones(real_x.size(0), dtype=torch.long, device=device), cfg.label_dim)
            fake_x = gen(z, cond).detach()
            fake_prob = disc(fake_x, cond)
            loss_fake = bce(fake_prob, torch.zeros_like(fake_prob))

            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()
            d_loss_epoch += d_loss.item() * real_x.size(0)

            # Train Generator
            opt_g.zero_grad()
            z = torch.randn(real_x.size(0), cfg.noise_dim, device=device)
            cond = one_hot(torch.ones(real_x.size(0), dtype=torch.long, device=device), cfg.label_dim)
            gen_x = gen(z, cond)
            gen_prob = disc(gen_x, cond)
            g_loss = bce(gen_prob, torch.ones_like(gen_prob))
            g_loss.backward()
            opt_g.step()
            g_loss_epoch += g_loss.item() * real_x.size(0)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log(f"Epoch {epoch+1}/{cfg.epochs}  D: {d_loss_epoch/n:.4f}  G: {g_loss_epoch/n:.4f}")

    return gen, disc


def generate_malignant_samples(gen: Generator, n_samples: int, cfg: GANConfig) -> np.ndarray:
    """Make n_samples malignant rows in feature space (scaled)."""
    device = torch.device(cfg.device)
    gen.eval()
    z = torch.randn(n_samples, cfg.noise_dim, device=device)
    cond = one_hot(torch.ones(n_samples, dtype=torch.long, device=device), cfg.label_dim)
    with torch.no_grad():
        synth = gen(z, cond).cpu().numpy()
    return synth


# 4) Train and test a plain classifier
def evaluate_logreg(X_train, y_train, X_test, y_test, description: str) -> dict:
    """Fit Logistic Regression. Report common metrics. Keep it simple."""
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED)),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    return {
        "setup": description,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def main():
    # Reset log
    if os.path.exists(OUT_LOG):
        os.remove(OUT_LOG)

    log("Start Cancer-GAN.")
    df = load_breast_cancer(DATA_PATH)
    log(f"Data shape: {df.shape}  Columns: {list(df.columns)}")

    # Label count helps sanity check.
    label_counts = df["diagnosis"].value_counts().to_dict()
    log(f"Label counts: {label_counts}")

    # Split
    X_train, X_test, y_train, y_test = make_splits(df, test_size=0.2)
    feature_cols = list(X_train.columns)
    log(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # Scale features for GAN
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Train GAN on malignant rows only
    cfg = GANConfig()
    cfg.device = "cpu"  # keep it simple and portable
    log("Train GAN (malignant only).")
    gen, disc = train_gan_on_malignant(X_train_scaled, y_train.values, cfg)
    log("GAN done.")

    # How many new malignant rows to make? Try to balance the train set.
    n_mal = int((y_train == 1).sum())
    n_ben = int((y_train == 0).sum())
    to_add = max(0, n_ben - n_mal)
    to_add = min(to_add, 1000)  # small safety cap
    log(f"Train benign: {n_ben}  malignant: {n_mal}  new malignant to add: {to_add}")

    # Make synthetic rows (still in scaled space)
    synth_scaled = generate_malignant_samples(gen, to_add, cfg)
    # Go back to the original feature scale
    synth_unscaled = scaler.inverse_transform(synth_scaled)

    # Build a small CSV with only synthetic malignant rows
    synth_df = pd.DataFrame(synth_unscaled, columns=feature_cols)
    synth_df.insert(0, "diagnosis", "M")  # store as 'M' for clarity
    synth_df.to_csv(OUT_AUG, index=False)
    log(f"Saved synthetic rows: {OUT_AUG}  shape={synth_df.shape}")

    # Make two train sets:
    # A) original
    Xtr_A = X_train.copy()
    ytr_A = y_train.copy()

    # B) GAN-augmented (append synth malignant rows)
    aug_train = pd.concat([
        pd.DataFrame({"diagnosis": ytr_A.map({0: "B", 1: "M"})}).assign(**{c: Xtr_A[c].values for c in feature_cols}),
        synth_df,
    ], ignore_index=True)
    ytr_B = aug_train["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    Xtr_B = aug_train.drop(columns=["diagnosis"])

    # Fit and score
    log("Score Logistic Regression on both setups.")
    results = []
    results.append(evaluate_logreg(Xtr_A.values, ytr_A.values, X_test.values, y_test.values, "Baseline (no aug)"))
    results.append(evaluate_logreg(Xtr_B.values, ytr_B.values, X_test.values, y_test.values, "GAN aug (malignant only)"))

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_METRICS, index=False)
    log(f"Saved metrics: {OUT_METRICS}\n{res_df}")

    log("Done. Script finished ok.")


if __name__ == "__main__":
    main()
