"""
LSTM - Prédiction du niveau des nappes phréatiques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# config globales

PIEZO_FILE = "time_series/piezos/BSS001CVLZ.csv"
FORCAGE_FILE = "time_series/forçages/A146020302.txt"
# MERGE_FILE = "dataset/A146020302_merged.csv"

SEQUENCE_LENGTH = 30 #30 jours d'entrée
FORECAST_HORIZON = 7 #7 jours de prédiction
HIDDEN_SIZE = 64
NUM_LAYERS = 2 #plus de layer c long 3 c faisable
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100 #100-200 pas plus
BATCH_SIZE = 32

features = ['Ptot', 'Temp', 'E_PM', 'Humi', 'Vent', 'IHGR', 'SWI'] #peut etre d'autre features meilleur je c pas trop
target = 'niveau_nappe_eau'

# on charge les données a partir de 2000-01-01

df_piezo = pd.read_csv(PIEZO_FILE)
df_piezo['date'] = pd.to_datetime(df_piezo['date_mesure'])
df_piezo = df_piezo[['date', 'niveau_nappe_eau']].set_index('date')

df_forcage = pd.read_csv(FORCAGE_FILE, sep=';', comment='#', skipinitialspace=True)
df_forcage.columns = df_forcage.columns.str.strip()
df_forcage['date'] = pd.to_datetime(df_forcage['Date'], format='%Y%m%d')
df_forcage = df_forcage.set_index('date')
df_forcage = df_forcage[features]

df = df_piezo.join(df_forcage, how='inner')
df = df[df.index >= '2000-01-01']
df = df.resample('D').mean().interpolate(method='linear', limit_direction='both')
df = df.replace(-99, np.nan).interpolate(method='linear', limit_direction='both').dropna()

# Version pour utiliser le merge file (version des deux fichiers combinés)

#df = pd.read_csv(MERGE_FILE)
#df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
#df = df.set_index('Date')
#df = df[features  + [target]]
#df = df[df.index >= '2000-01-01']
#df = df.resample('D').mean().interpolate(method='linear', limit_direction='both')
#df = df.replace(-99, np.nan).interpolate(method='linear', limit_direction='both').dropna()

# on prépare les données

X = df[features].values
y = df[target].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

def create_sequences(X, y, seq_length, forecast_horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length - forecast_horizon + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH, FORECAST_HORIZON)

train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# construction du modèle LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# le training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train.squeeze(-1)).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test.squeeze(-1)).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMModel(len(features), HIDDEN_SIZE, NUM_LAYERS, FORECAST_HORIZON, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_test_t)
        val_loss = criterion(val_output, y_test_t).item()
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {val_loss:.6f}")

model.load_state_dict(torch.load('best_lstm_model.pth'))

# rolling forecast bien pour des séries temporelles 

model.eval()
test_start_idx = train_size + SEQUENCE_LENGTH

all_predictions = []
all_actuals = []
all_dates = []

current_idx = test_start_idx

while current_idx + FORECAST_HORIZON <= len(df):
    seq_start = current_idx - SEQUENCE_LENGTH
    sequence = X_scaled[seq_start:current_idx]
    
    with torch.no_grad():
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        pred_scaled = model(seq_tensor).cpu().numpy()[0]
    
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = y[current_idx:current_idx + FORECAST_HORIZON].flatten()
    dates = df.index[current_idx:current_idx + FORECAST_HORIZON]
    
    all_predictions.append(pred)
    all_actuals.append(actual)
    all_dates.append(dates)
    
    current_idx += FORECAST_HORIZON

# Reconstruction série temporelle
dates_flat = []
preds_flat = []
actuals_flat = []

for dates, preds, acts in zip(all_dates, all_predictions, all_actuals):
    for d, p, a in zip(dates, preds, acts):
        dates_flat.append(d)
        preds_flat.append(p)
        actuals_flat.append(a)

# la métrique

rmse = np.sqrt(mean_squared_error(actuals_flat, preds_flat))
mae = mean_absolute_error(actuals_flat, preds_flat)
r2 = r2_score(actuals_flat, preds_flat)

print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# graph de fin

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(dates_flat, actuals_flat, 'b-', linewidth=1, label='Réel')
ax.plot(dates_flat, preds_flat, 'r-', linewidth=1, label='Prédit (rolling)')

ax.set_xlabel('Date')
ax.set_ylabel('Niveau nappe (m)')
ax.set_title('Rolling Forecast - Comparaison Réel vs Prédit')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

