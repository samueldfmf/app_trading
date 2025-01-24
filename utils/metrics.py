import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_tensor, y_tensor, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
        targets = y_tensor.cpu().numpy()
    
    # Calcular métricas
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    smape = np.mean(2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + 1e-10)) * 100
    r2 = r2_score(targets, predictions)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SMAPE": smape,
        "R²": r2,
    }
    
    return predictions, metrics
