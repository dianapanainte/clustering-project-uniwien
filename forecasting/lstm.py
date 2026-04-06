import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CONFIG = {
    "csv_path":       "sample_23.csv",
    "id_column":      "ID",
    "seq_len":        60,
    "pred_len":       60,
    "hidden_size":    128,
    "num_layers":     2,
    "dropout":        0.2,
    "batch_size":     64,
    "epochs":         50,
    "lr":             1e-3,
    "val_split":      0.1,
    "seed":           42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def load_data(csv_path: str, id_col: str) -> tuple[np.ndarray, list, StandardScaler]:
    df = pd.read_csv(csv_path)
    ids = df[id_col].tolist()
    values = df.drop(columns=[id_col]).values.astype(np.float32)

    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values.T).T

    return values_scaled, ids, scaler

class EnergyDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.X, self.y = [], []

        n_households, n_days = data.shape
        max_start = n_days - seq_len - pred_len + 1

        for h in range(n_households):
            for start in range(max_start):
                x = data[h, start : start + seq_len]
                y = data[h, start + seq_len : start + seq_len + pred_len]
                self.X.append(x)
                self.y.append(y)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 dropout=0.2, pred_len=366):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


def train(model, loader, val_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    print(f"\nBest Val MSE: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    return model


def predict_2024(model, data_scaled: np.ndarray, cfg) -> np.ndarray:
    model.eval()
    seq_len = cfg["seq_len"]
    seed    = data_scaled[:, -seq_len:]
    seed_t  = torch.tensor(seed, dtype=torch.float32) \
                   .unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        preds = model(seed_t).cpu().numpy()
    return preds

def main():
    cfg = CONFIG

    # Load
    print("Loading data...")
    data_scaled, ids, scaler = load_data(cfg["csv_path"], cfg["id_column"])
    n_households, n_days = data_scaled.shape
    print(f"  {n_households} households | {n_days} days of history")

    n_val  = max(1, int(n_households * cfg["val_split"]))
    n_train = n_households - n_val
    train_data = data_scaled[:n_train]
    val_data   = data_scaled[n_train:]

    val_seq_days = n_days - cfg["pred_len"] if n_days > cfg["pred_len"] else n_days // 2

    train_ds = EnergyDataset(train_data, cfg["seq_len"], cfg["pred_len"])
    val_ds   = EnergyDataset(val_data[:, :val_seq_days], cfg["seq_len"],
                              min(cfg["pred_len"], val_seq_days - cfg["seq_len"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = LSTMForecaster(
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        pred_len    = cfg["pred_len"],
    ).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    model = train(model, train_loader, val_loader, cfg)

    torch.save(model.state_dict(), "lstm_energy.pt")
    print("Model saved to lstm_energy.pt")

    print("\nPredicting 2024...")
    preds_scaled = predict_2024(model, data_scaled, cfg)

    preds_original = scaler.inverse_transform(preds_scaled.T).T

    dates_2024 = pd.date_range("2024-01-01", periods=366, freq="D").strftime("%Y-%m-%d").tolist()

    out_df = pd.DataFrame(preds_original, columns=dates_2024)
    out_df.insert(0, cfg["id_column"], ids)
    out_df.to_csv("forecast_lstm_2024.csv", index=False)
    print(f"Predictions saved to forecast_lstm_2024.csv  ({out_df.shape})")

    print("\nSample predictions (first 3 households, first 7 days of 2024):")
    print(out_df.iloc[:3, :8].to_string(index=False))


if __name__ == "__main__":
    main()
