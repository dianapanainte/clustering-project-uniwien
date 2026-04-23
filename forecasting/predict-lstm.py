"""
Optional args:
    --lookback      Number of past days used as input window  (default: 30)
    --hidden        LSTM hidden size                           (default: 128)
    --layers        Number of LSTM layers                     (default: 2)
    --epochs        Training epochs                           (default: 50)
    --batch         Batch size                                (default: 256)
    --lr            Learning rate                             (default: 1e-3)
    --val_split     Fraction of households held out for val   (default: 0.1)
    --dropout       Dropout rate (applied if layers > 1)      (default: 0.2)
    --seed          Random seed                               (default: 42)
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    p = argparse.ArgumentParser(description="LSTM energy forecaster")
    p.add_argument("--input",    required=True,  help="Path to input CSV")
    p.add_argument("--output",   default="forecast_24_lstm.csv")
    p.add_argument("--lookback", type=int,   default=30)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--layers",   type=int,   default=2)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--val_split",type=float, default=0.1)
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    print(f"\n[1/5] Loading data from '{path}' ...")
    df = pd.read_csv(path)

    id_col = df.columns[0]
    ids    = df[id_col].tolist()
    data   = df.iloc[:, 1:].values.astype(np.float32)

    print(f"      Households : {data.shape[0]}")
    print(f"      Days       : {data.shape[1]}")

    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1, keepdims=True) + 1e-8
    data_norm = (data - mean) / std

    scaler = {"mean": mean.squeeze(1), "std": std.squeeze(1)}
    return ids, data, data_norm, scaler


class SlidingWindowDataset(Dataset):
    def __init__(self, data_norm: np.ndarray, lookback: int):
        self.lookback = lookback
        xs, ys = [], []
        N, T = data_norm.shape
        for i in range(N):
            series = data_norm[i]
            for t in range(T - lookback):
                xs.append(series[t : t + lookback])
                ys.append(series[t + lookback])
        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.Y = torch.tensor(np.array(ys), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.Y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_model(data_norm, args, device):
    N = data_norm.shape[0]

    indices = np.arange(N)
    np.random.shuffle(indices)
    val_n   = max(1, int(N * args.val_split))
    val_idx = indices[:val_n]
    trn_idx = indices[val_n:]

    trn_dataset = SlidingWindowDataset(data_norm[trn_idx], args.lookback)
    val_dataset = SlidingWindowDataset(data_norm[val_idx], args.lookback)

    trn_loader = DataLoader(trn_dataset, batch_size=args.batch, shuffle=True,
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"\n[2/5] Dataset split:")
    print(f"      Train households : {len(trn_idx)}  |  samples : {len(trn_dataset)}")
    print(f"      Val   households : {len(val_idx)}  |  samples : {len(val_dataset)}")

    model = LSTMForecaster(
        hidden_size = args.hidden,
        num_layers  = args.layers,
        dropout     = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3/5] Model parameters : {total_params:,}")
    print(f"      Device           : {device}")
    if device.type == "cuda":
        print(f"      GPU              : {torch.cuda.get_device_name(0)}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, verbose=True)

    best_val_loss = float("inf")
    best_state    = None

    print(f"\n[4/5] Training for {args.epochs} epochs ...\n")
    header = f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        trn_loss = 0.0
        for xb, yb in trn_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item() * len(xb)
        trn_loss /= len(trn_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred    = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {trn_loss:>12.6f}  {val_loss:>10.6f}  {elapsed:>8.1f}")

    print(f"\n      Best val loss : {best_val_loss:.6f}")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def forecast_all(model, data_norm, scaler, args, device, forecast_steps=366):
    print(f"\n[5/5] Generating 366-day forecasts for all households ...")
    model.eval()
    N        = data_norm.shape[0]
    preds_all = np.zeros((N, forecast_steps), dtype=np.float32)

    for i in range(N):
        window = list(data_norm[i, -args.lookback:])
        preds  = []
        for _ in range(forecast_steps):
            x   = torch.tensor(window[-args.lookback:], dtype=torch.float32)
            x   = x.unsqueeze(0).unsqueeze(-1).to(device)
            out = model(x).item()
            preds.append(out)
            window.append(out)

        mean_i = scaler["mean"][i]
        std_i  = scaler["std"][i]
        preds_all[i] = np.array(preds) * std_i + mean_i

    return preds_all


def save_forecast(ids, preds_all, output_path: str):
    forecast_steps = preds_all.shape[1]
    dates = pd.date_range(start="2024-01-01", periods=forecast_steps, freq="D").strftime("%Y-%m-%d").tolist()
    cols  = ["ID"] + dates
    rows  = [[ids[i]] + preds_all[i].tolist() for i in range(len(ids))]
    df_out = pd.DataFrame(rows, columns=cols)
    df_out.to_csv(output_path, index=False)
    print(f"Saved forecast in '{output_path}'")
    print(f"Shape: {df_out.shape} (households x (1 + 366) cols)")


def main():
    args   = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not found - running on CPU")

    ids, data, data_norm, scaler = load_data(args.input)

    model     = train_model(data_norm, args, device)
    preds_all = forecast_all(model, data_norm, scaler, args, device)

    save_forecast(ids, preds_all, args.output)

    weights_path = Path(args.output).with_suffix(".pt")
    torch.save(model.state_dict(), weights_path)
    print(f"      Model weights  → '{weights_path}'")
    print("\n✓ Done.\n")


if __name__ == "__main__":
    main()
