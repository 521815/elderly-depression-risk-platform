"""Small temporal deep model on raw MPDD audio sequences (proof of concept)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold

TIME_LEN = 64
PCA_DIM = 32


class TinyCNN(nn.Module):
    def __init__(self, in_dim: int = PCA_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(48 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def resample_seq(arr: np.ndarray, length: int) -> np.ndarray:
    idx = np.linspace(0, len(arr) - 1, length).astype(int)
    return arr[idx]


def main() -> None:
    root = PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder"
    labels = pd.read_csv(root / "split_labels_train.csv")
    labels = labels.rename(columns={"ID": "subject_id"})
    labels["subject_id"] = labels["subject_id"].astype(str)
    labels["label"] = (labels["PHQ-9"] >= 8).astype(int)

    rows = []
    for path in sorted((root / "Audio" / "train" / "wav2vec2").rglob("*.npy")):
        subject = str(int(path.parent.name))
        arr = np.load(path, allow_pickle=False).astype("float32")
        rows.append(
            {
                "subject_id": subject,
                "seq": resample_seq(arr, TIME_LEN),
            }
        )
    seq_df = pd.DataFrame(rows)
    df = seq_df.merge(labels, on="subject_id", how="inner")
    if len(df) < 20:
        raise SystemExit("可用原始音频序列不足")

    X = np.stack(df["seq"].to_numpy())  # (n, time, features)
    flat = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=min(PCA_DIM, flat.shape[1], flat.shape[0] - 1))
    pca.fit(flat)
    X_pca = np.stack(
        [pca.transform(x).astype("float32") for x in X]
    )  # (n, time, pca_dim)
    X_pca = X_pca.transpose(0, 2, 1)  # (n, pca_dim, time)
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_pca, y, groups=groups)):
        device = "cpu"
        model = TinyCNN(in_dim=X_pca.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        X_tr_t = torch.tensor(X_pca[tr_idx], dtype=torch.float32)
        y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32)
        model.train()
        for _ in range(3):
            opt.zero_grad()
            logits = model(X_tr_t)
            loss = loss_fn(logits, y_tr_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_prob = torch.sigmoid(model(torch.tensor(X_pca[tr_idx], dtype=torch.float32))).numpy()
            te_prob = torch.sigmoid(model(torch.tensor(X_pca[te_idx], dtype=torch.float32))).numpy()
        tr_sub = pd.Series(tr_prob, index=groups[tr_idx]).groupby(level=0).mean()
        te_sub = pd.Series(te_prob, index=groups[te_idx]).groupby(level=0).mean()
        y_tr_sub = pd.Series(y[tr_idx], index=groups[tr_idx]).groupby(level=0).first()
        y_te_sub = pd.Series(y[te_idx], index=groups[te_idx]).groupby(level=0).first()
        threshold = best_threshold(y_tr_sub.to_numpy(), tr_sub.to_numpy(), metric="f1")
        pred = (te_sub.to_numpy() >= threshold).astype(int)
        aucs.append(roc_auc_score(y_te_sub.to_numpy(), te_sub.to_numpy()))
        f1s.append(f1_score(y_te_sub.to_numpy(), pred, zero_division=0))
        print(f"fold {fold} auc={aucs[-1]:.3f} f1={f1s[-1]:.3f}")

    result = {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
        "rows": int(len(df)),
        "model": "TinyCNN audio wav2vec2",
        "time_len": TIME_LEN,
        "pca_dim": PCA_DIM,
        "note": "原始音频时序建模可行性验证：PCA 降维 + 定长 + 小型 1D CNN，subject-aware 5 折",
    }
    out = PROJECT_ROOT / "output" / "temporal_deep_report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(
        PROJECT_ROOT / "output" / "temporal_deep_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
