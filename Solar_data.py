# Solar Irradiance Forecasting with AI (LSTM, BPNN, LR, Persistence)
# This script replicates the study by Qing and Niu (2018) for SDG-aligned solar energy forecasting

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tabulate import tabulate

# === Constants ===
GHI_MIN, GHI_MAX = 0, 1150  # Physical irradiance range (W/m²)

# === Utility Functions ===
def load_data(path):
    """Load data from CSV and split into features and targets."""
    data = np.loadtxt(path, delimiter=',')
    x, t = data[:, :9], data[:, -1]
    return x, t

def reshape_for_sequence(x, t):
    """Reshape hourly data into daily sequences for sequence models like LSTM."""
    n_days = x.shape[0] // 11
    return x[:n_days*11].reshape(n_days, 11, 9), t[:n_days*11].reshape(n_days, 11, 1)

def denormalize(y):
    """Convert [-1,1] normalized values to physical scale."""
    return (y + 1) * (GHI_MAX - GHI_MIN) / 2 + GHI_MIN

# === Load and Reshape Datasets ===
train_x_raw, train_t_raw = load_data(r"C:\Users\abdir\OneDrive\Desktop\AI_For_Global_Challanges\train_NREL_solar_data.csv")
val_x_raw, val_t_raw = load_data(r"C:\Users\abdir\OneDrive\Desktop\AI_For_Global_Challanges\validate_NREL_solar_data.csv")
test_x_raw, test_t_raw = load_data(r"C:\Users\abdir\OneDrive\Desktop\AI_For_Global_Challanges\test_NREL_solar_data.csv")

train_x, train_t = reshape_for_sequence(train_x_raw, train_t_raw)
val_x, val_t = reshape_for_sequence(val_x_raw, val_t_raw)
test_x, test_t = reshape_for_sequence(test_x_raw, test_t_raw)

# === 1. LSTM Model ===
lstm_model = Sequential([
    LSTM(30, return_sequences=True, input_shape=(11, 9)),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the LSTM model
lstm_history = lstm_model.fit(
    train_x, train_t,
    validation_data=(val_x, val_t),
    epochs=100,
    batch_size=32,
    verbose=0
)

# Predict with LSTM
lstm_pred = lstm_model.predict(test_x).reshape(-1)
lstm_rmse = sqrt(mean_squared_error(test_t.reshape(-1), lstm_pred)) * (GHI_MAX / 2)

# === 2. Persistence Model ===
test_t_day = test_t.reshape(-1, 11)
persistence_pred = np.vstack([test_t_day[0], test_t_day[:-1]])  # Use previous day's irradiance
rmse_persistence = sqrt(mean_squared_error(denormalize(test_t_day).flatten(),
                                           denormalize(persistence_pred).flatten()))

# === 3. Linear Regression Model ===
X_train_lr = train_x.reshape(-1, 9)
y_train_lr = train_t.reshape(-1)
X_test_lr = test_x.reshape(-1, 9)
y_test_lr = test_t.reshape(-1)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
lr_pred = lr_model.predict(X_test_lr)
lr_pred_phys = denormalize(lr_pred)
y_test_phys = denormalize(y_test_lr)
lr_rmse = sqrt(mean_squared_error(y_test_phys, lr_pred_phys))

# === 4. BPNN Model ===
train_x_bpnn = train_x_raw.reshape(-1, 99)
val_x_bpnn = val_x_raw.reshape(-1, 99)
test_x_bpnn = test_x_raw.reshape(-1, 99)
train_t_bpnn = train_t_raw.reshape(-1, 11)
val_t_bpnn = val_t_raw.reshape(-1, 11)
test_t_bpnn = test_t_raw.reshape(-1, 11)

bpnn_model = Sequential([
    Dense(50, activation='tanh', input_shape=(99,)),
    Dense(11)
])
bpnn_model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
bpnn_history = bpnn_model.fit(
    train_x_bpnn, train_t_bpnn,
    validation_data=(val_x_bpnn, val_t_bpnn),
    epochs=2500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate BPNN
bpnn_pred = bpnn_model.predict(test_x_bpnn)
bpnn_rmse = sqrt(mean_squared_error(denormalize(test_t_bpnn).flatten(),
                                    denormalize(bpnn_pred).flatten()))

# === Table: Performance Comparison ===
table_data = [
    ["Persistence", rmse_persistence, "209.25", f"{(rmse_persistence - 209.25) / 209.25 * 100:.1f}%"],
    ["Linear Regression", lr_rmse, "230.99", f"{(lr_rmse - 230.99) / 230.99 * 100:.1f}%"],
    ["BPNN", bpnn_rmse, "133.53", f"{(bpnn_rmse - 133.53) / 133.53 * 100:.1f}%"],
    ["LSTM", lstm_rmse, "76.25", f"{(lstm_rmse - 76.25) / 76.25 * 100:.1f}%"]
]

print("\nComparison of Model Performance (Matching Paper Table 4):")
print(tabulate(table_data, headers=["Model", "Your RMSE (W/m²)", "Paper RMSE (W/m²)", "Difference"], tablefmt="grid"))

# === Table: Improvement Metrics ===
improvement_data = [
    ["LSTM vs BPNN", f"{((bpnn_rmse - lstm_rmse) / bpnn_rmse * 100):.1f}%", "42.9%"],
    ["LSTM vs LR", f"{((lr_rmse - lstm_rmse) / lr_rmse * 100):.1f}%", "61.1%"],
    ["LSTM vs Persistence", f"{((rmse_persistence - lstm_rmse) / rmse_persistence * 100):.1f}%", "63.6%"]
]

print("\nImprovement Over Baselines:")
print(tabulate(improvement_data, headers=["Comparison", "Your Improvement", "Paper Improvement"], tablefmt="grid"))

# === Visualization: LSTM Loss Plot ===
plt.figure(figsize=(8, 5))
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Visualization: BPNN Loss Plot ===
plt.figure(figsize=(8, 5))
plt.plot(bpnn_history.history['loss'], label='BPNN Train Loss')
plt.plot(bpnn_history.history['val_loss'], label='BPNN Val Loss')
plt.title("BPNN Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("✅ Finished LSTM training")
print("✅ Finished BPNN training")
