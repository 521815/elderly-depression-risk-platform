"""Improved temporal deep model on raw MPDD audio sequences.

The v1 TinyCNN fitted PCA on all data and trained for only three epochs.
This variant fits preprocessing per training fold, adds normalization,
dropout, a bidirectional GRU with attention, and uses class-weighted loss.
"""

from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold

class TemporalAudioNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        hidden: int = 64,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(min(dropout, 0.3)),
            nn.AdaptiveAvgPool1d(64),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(min(dropout, 0.3)),
            nn.AdaptiveAvgPool1d(32),
        )
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.05, 0.2)),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, feature, time)
        h = self.conv(x).transpose(1, 2)  # (B, pooled_time, channels)
        out, _ = self.gru(h)  # (B, pooled_time, 2*hidden)
        attn = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        ctx = (out * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx).squeeze(-1)


class ConvPoolNet(nn.Module):
    def __init__(self, in_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(min(dropout, 0.3)),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def resample_seq(arr: np.ndarray, length: int) -> np.ndarray:
    idx = np.linspace(0, len(arr) - 1, length).astype(int)
    return arr[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-len", type=int, default=128)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--arch", choices=["gru", "conv"], default="gru")
    parser.add_argument("--tag", default="final")
    args = parser.parse_args()

    root = PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder"
    labels = pd.read_csv(root / "split_labels_train.csv")
    labels = labels.rename(columns={"ID": "subject_id"})
    labels["subject_id"] = labels["subject_id"].astype(str)
    labels["label"] = (labels["PHQ-9"] >= 8).astype(int)

    rows = []
    for path in sorted((root / "Audio" / "train" / "wav2vec2").rglob("*.npy")):
        subject = str(int(path.parent.name))
        arr = np.load(path, allow_pickle=False).astype("float32")
        rows.append({"subject_id": subject, "seq": resample_seq(arr, args.time_len)})
    seq_df = pd.DataFrame(rows)
    df = seq_df.merge(labels, on="subject_id", how="inner")
    if len(df) < 20:
        raise SystemExit("可用原始音频序列不足")

    X_raw = np.stack(df["seq"].to_numpy())
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()
    print("rows", len(df), "subjects", pd.Series(groups).nunique(), "positive", int(y.sum()))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_raw, y, groups=groups)):
        tr_flat = X_raw[tr_idx].reshape(-1, X_raw.shape[-1])
        te_flat = X_raw[te_idx].reshape(-1, X_raw.shape[-1])
        scaler = StandardScaler()
        tr_flat = scaler.fit_transform(tr_flat).astype("float32")
        te_flat = scaler.transform(te_flat).astype("float32")
        pca = PCA(
            n_components=min(args.pca_dim, tr_flat.shape[1], len(tr_idx) - 1),
            random_state=42,
        )
        tr_pca = pca.fit_transform(tr_flat).astype("float32")
        te_pca = pca.transform(te_flat).astype("float32")
        X_tr = np.stack(tr_pca.reshape(len(tr_idx), args.time_len, -1)).transpose(0, 2, 1)
        X_te = np.stack(te_pca.reshape(len(te_idx), args.time_len, -1)).transpose(0, 2, 1)

        pos = int(y[tr_idx].sum())
        neg = int(len(tr_idx) - pos)
        pos_weight = torch.tensor([max(neg, 1) / max(pos, 1)], dtype=torch.float32)
        torch.manual_seed(42 + fold)
        if args.arch == "conv":
            model = ConvPoolNet(in_dim=X_tr.shape[1], dropout=args.dropout)
        else:
            model = TemporalAudioNet(
                in_dim=X_tr.shape[1],
                hidden=args.hidden,
                dropout=args.dropout,
            )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=args.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y[tr_idx], dtype=torch.float32),
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42 + fold),
        )
        model.train()
        for epoch in range(args.epochs):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.item())
            if epoch % 10 == 0:
                print(f"fold {fold} epoch {epoch} loss {total / len(loader):.4f}")

        model.eval()
        with torch.inference_mode():
            tr_prob = torch.sigmoid(model(torch.tensor(X_tr, dtype=torch.float32))).numpy()
            te_prob = torch.sigmoid(model(torch.tensor(X_te, dtype=torch.float32))).numpy()
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
        "model": f"{args.arch} audio wav2vec2 ({args.tag})",
        "time_len": args.time_len,
        "pca_dim": args.pca_dim,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "note": "逐折标准化+PCA、双向 GRU+注意力、类别加权、subject-aware 5 折",
    }
    out = PROJECT_ROOT / "output" / f"temporal_deep_{args.tag}_report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(
        PROJECT_ROOT / "output" / f"temporal_deep_{args.tag}_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
