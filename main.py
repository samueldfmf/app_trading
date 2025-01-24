from data.preprocessing import prepare_data
from model.lstm_model import LSTMWithAttention
from model.model_utils import save_model, load_model
from utils.metrics import evaluate_model
from visualizations.plots import plot_predictions

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ejecutando en: {device}")

# Configuración
LOOK_BACK = 60
BATCH_SIZE = 32
EPOCHS = 50
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0005
MODEL_PATH = "lstm_model_v2.pth"

# Preparar datos
data, scaler = prepare_data(include_today=True, look_back=LOOK_BACK)
X, y = data["X"], data["y"]

# Si los datos no son suficientes
if len(X) < 2:
    print("No hay suficientes datos para entrenamiento y prueba.")
    exit()

# Convertir a tensores
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
train_data = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Definir modelo
model = LSTMWithAttention(input_size=X.shape[2], hidden_layer_size=100, output_size=1, dropout=DROPOUT_RATE).to(device)
load_model(model, MODEL_PATH)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época {epoch + 1}/{EPOCHS}, Pérdida: {total_loss / len(train_loader)}")

# Guardar el modelo
save_model(model, MODEL_PATH)

# Evaluación
predictions, metrics = evaluate_model(model, X_tensor, y_tensor, scaler)

# Último precio real y predicción futura
last_close_vector = np.zeros(X.shape[2])
last_close_vector[0] = X[-1, -1, 0]  # Último valor normalizado
last_close_price = scaler.inverse_transform([last_close_vector])[0][0]  # Escalar a precio original

future_data = torch.tensor(X[-1:], dtype=torch.float32).to(device)
with torch.no_grad():
    future_prediction = model(future_data).cpu().numpy()[0, 0]

future_prediction_price = scaler.inverse_transform([[future_prediction] + [0] * (X.shape[2] - 1)])[0][0]

# Mensajes
print(f"El último precio real utilizado es: {last_close_price}")
print(f"La predicción del próximo precio es: {future_prediction_price}")

# Imprimir métricas
print("Métricas del modelo:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# Visualización
plot_predictions(predictions, y_tensor.cpu().numpy())
