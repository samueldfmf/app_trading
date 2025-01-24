import matplotlib.pyplot as plt

def plot_predictions(predictions, targets):
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label="Valores Reales")
    plt.plot(predictions, label="Predicciones", linestyle="--")
    plt.title("Valores Reales vs Predicciones")
    plt.xlabel("Índice")
    plt.ylabel("Precio de Cierre")
    plt.legend()
    plt.show()
