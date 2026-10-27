"""Inspecciona, limpia y modela los datos de clientes del grupo 10."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "graficos"
INPUT_PATH = DATA_DIR / "grupo_10.csv"
CLEAN_PATH = DATA_DIR / "grupo_10_limpio.csv"
FEATURES = ["visitas_mes", "ticket_promedio_miles"]
TARGET = "cliente_recurrente"


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Calcula la sigmoide limitando los logits para evitar desbordamientos."""
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_probabilities(
    features_scaled: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    return sigmoid(features_scaled @ weights + bias)


def binary_log_loss(actual: np.ndarray, probabilities: np.ndarray) -> float:
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(
            actual * np.log(probabilities)
            + (1.0 - actual) * np.log(1.0 - probabilities)
        )
    )


def accuracy(actual: np.ndarray, probabilities: np.ndarray) -> float:
    predicted = (probabilities >= 0.5).astype(int)
    return float(np.mean(predicted == actual))


def inspect_data(raw: pd.DataFrame) -> None:
    print("\n=== INSPECCIÓN DEL CSV ORIGINAL ===")
    print("Primeras cinco filas:")
    print(raw.head().to_string(index=False))
    print(f"\nForma del DataFrame: {raw.shape}")
    print("\nTipos de datos:")
    print(raw.dtypes.to_string())
    print("\nValores nulos por columna:")
    print(raw.isna().sum().to_string())
    print(f"\nFilas completamente duplicadas: {raw.duplicated().sum()}")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    print("\n=== LIMPIEZA Y EVIDENCIA ===")
    print(f"Filas originales, incluido pie de tabla: {len(raw)}")
    summary_mask = raw["cliente_id"].astype("string").str.strip().str.upper().eq(
        "TOTAL"
    )
    summary_rows = raw.loc[summary_mask]
    print(f"Filas de resumen detectadas por cliente_id == 'TOTAL': {len(summary_rows)}")
    if not summary_rows.empty:
        print(summary_rows.to_string(index=False))

    clients = raw.loc[~summary_mask].copy()
    print(f"Filas de clientes después de excluir el resumen: {len(clients)}")
    duplicate_count = int(clients.duplicated().sum())
    duplicate_ids = int(clients["cliente_id"].duplicated().sum())
    print(f"Filas de clientes duplicadas: {duplicate_count}")
    print(f"Identificadores de cliente duplicados: {duplicate_ids}")
    if duplicate_count or duplicate_ids:
        raise ValueError("Se detectaron duplicados y requieren revisión explícita.")

    missing_visits = clients["visitas_mes"].isna()
    missing_count = int(missing_visits.sum())
    median_visits = float(clients["visitas_mes"].median())
    print(f"Visitas faltantes: {missing_count}; mediana observada: {median_visits:g}")
    if missing_count:
        print(
            "Clientes con visitas faltantes: "
            f"{clients.loc[missing_visits, 'cliente_id'].tolist()}"
        )
        clients.loc[missing_visits, "visitas_mes"] = median_visits

    outliers = clients.loc[clients["ticket_promedio_miles"] > 1000]
    print(
        "Tickets > 1000 miles de pesos: "
        f"{outliers[['cliente_id', 'ticket_promedio_miles']].to_dict('records')}"
    )
    tickets_without_outliers = clients.loc[
        ~clients.index.isin(outliers.index), "ticket_promedio_miles"
    ]
    print(
        "Resumen de los demás tickets (mínimo, mediana, máximo): "
        f"{tickets_without_outliers.min():g}, "
        f"{tickets_without_outliers.median():g}, "
        f"{tickets_without_outliers.max():g}"
    )
    suspicious_mean_before = float(clients["ticket_promedio_miles"].mean())
    corrections = {"K058": (191000.0, 191.0)}
    for customer_id, (observed, corrected) in corrections.items():
        matching = clients["cliente_id"].eq(customer_id)
        if int(matching.sum()) != 1:
            raise ValueError(f"No se encontró un único registro {customer_id}.")
        current = float(clients.loc[matching, "ticket_promedio_miles"].iloc[0])
        if current != observed or int(outliers["cliente_id"].eq(customer_id).sum()) != 1:
            raise ValueError(
                f"El valor sospechoso de {customer_id} cambió; "
                "no se corregirá sin volver a revisar la evidencia."
            )
        clients.loc[matching, "ticket_promedio_miles"] = corrected
        print(
            f"Corrección documentada: {customer_id}, ticket "
            f"{observed:g} -> {corrected:g}."
        )
    suspicious_mean_after = float(clients["ticket_promedio_miles"].mean())

    category_spellings = clients["barrio"].dropna().unique().tolist()
    normalized_categories = {
        str(value).strip().casefold() for value in category_spellings
    }
    print(f"Escrituras originales de barrio: {category_spellings}")
    print(f"Categorías únicas después de normalizar: {sorted(normalized_categories)}")
    clients["barrio"] = clients["barrio"].map(
        lambda value: str(value).strip().title()
    )

    if clients[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Quedaron faltantes en variables del modelo.")
    if not clients[TARGET].isin([0, 1]).all():
        raise ValueError("El objetivo debe contener únicamente 0 y 1.")
    if (clients[FEATURES] < 0).any().any():
        raise ValueError("Se encontró un valor negativo que requiere revisión.")

    clients[TARGET] = clients[TARGET].astype(int)
    clients["visitas_mes"] = clients["visitas_mes"].astype(int)
    clients["ticket_promedio_miles"] = clients[
        "ticket_promedio_miles"
    ].astype(int)
    clients = clients.reset_index(drop=True)
    clients.to_csv(CLEAN_PATH, index=False)
    class_counts = clients[TARGET].value_counts().sort_index().to_dict()

    print(f"\nMedia de ticket antes de corregir K058: {suspicious_mean_before:.4f}")
    print(f"Media de ticket después de corregir K058: {suspicious_mean_after:.4f}")
    print(f"Filas de clientes antes de limpiar: {len(raw) - len(summary_rows)}")
    print(f"Filas limpias guardadas: {len(clients)}")
    print(f"Registros de clase 0: {class_counts.get(0, 0)}")
    print(f"Registros de clase 1: {class_counts.get(1, 0)}")
    print(f"CSV limpio: {CLEAN_PATH}")

    evidence: dict[str, object] = {
        "summary_rows": len(summary_rows),
        "missing_visits": missing_count,
        "median_visits": median_visits,
        "category_spellings": category_spellings,
        "suspicious_mean_before": suspicious_mean_before,
        "suspicious_mean_after": suspicious_mean_after,
        "class_counts": class_counts,
    }
    return clients, evidence


def create_eda_plots(data: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_data = data.assign(
        tipo_cliente=data[TARGET].map({0: "Ocasional (0)", 1: "Recurrente (1)"})
    )
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue="tipo_cliente",
        hue_order=["Ocasional (0)", "Recurrente (1)"],
        palette={"Ocasional (0)": "#4472C4", "Recurrente (1)": "#E67E22"},
        s=75,
    )
    plt.title("Visitas y ticket promedio por tipo de cliente")
    plt.xlabel("Visitas al mes")
    plt.ylabel("Ticket promedio (miles de pesos)")
    plt.legend(title="Cliente recurrente")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dispersion_clientes.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x=TARGET, y=FEATURES[0], hue=TARGET, legend=False)
    plt.title("Visitas mensuales por clase de cliente")
    plt.xlabel("Cliente recurrente (0 = ocasional, 1 = recurrente)")
    plt.ylabel("Visitas al mes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplot_visitas_clase.png", dpi=160)
    plt.close()


def train_perceptron(data: pd.DataFrame) -> dict[str, object]:
    features = data[FEATURES].to_numpy(dtype=float)
    target = data[TARGET].to_numpy(dtype=float)
    means = features.mean(axis=0)
    standard_deviations = features.std(axis=0)
    if np.any(standard_deviations == 0):
        raise ValueError("Una entrada tiene desviación estándar igual a cero.")
    scaled = (features - means) / standard_deviations

    weights = np.zeros(scaled.shape[1], dtype=float)
    bias = 0.0
    initial_weights = weights.copy()
    initial_bias = bias
    initial_loss = binary_log_loss(
        target, predict_probabilities(scaled, weights, bias)
    )
    learning_rate = 0.1
    epochs = 1000
    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(epochs):
        probabilities = predict_probabilities(scaled, weights, bias)
        error = probabilities - target
        weights -= learning_rate * (scaled.T @ error) / len(target)
        bias -= learning_rate * float(error.mean())
        updated_probabilities = predict_probabilities(scaled, weights, bias)
        losses.append(binary_log_loss(target, updated_probabilities))
        accuracies.append(accuracy(target, updated_probabilities))

    final_probabilities = predict_probabilities(scaled, weights, bias)
    final_loss = losses[-1]
    final_accuracy = accuracies[-1]
    mistakes = int(np.sum((final_probabilities >= 0.5) != target))
    print("\n=== PERCEPTRÓN NUMPY ===")
    print(f"Entradas: {FEATURES}; filas usadas para entrenar: {len(data)}")
    print(f"Media de entrenamiento [visitas, ticket]: {means}")
    print(f"Desviación de entrenamiento [visitas, ticket]: {standard_deviations}")
    print(
        "Pesos iniciales (w1, w2, b): "
        f"({initial_weights[0]:.6f}, {initial_weights[1]:.6f}, "
        f"{initial_bias:.6f})"
    )
    print(
        f"Pérdida inicial: {initial_loss:.6f}; "
        f"pérdida final ({epochs} épocas): {final_loss:.6f}"
    )
    print(
        "Pesos finales (w1, w2, b): "
        f"({weights[0]:.6f}, {weights[1]:.6f}, {bias:.6f})"
    )
    print(f"Exactitud final en los datos de entrenamiento: {final_accuracy:.4f}")
    print(
        f"Registros del conjunto completo al lado equivocado "
        f"(umbral 0.5, evaluación en entrenamiento): {mistakes}/{len(data)}"
    )
    print(f"Pérdida y exactitud registradas: {len(losses)} épocas.")

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), losses, color="#4472C4")
    plt.title("Curva de aprendizaje del perceptrón")
    plt.xlabel("Época")
    plt.ylabel("Pérdida logarítmica")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "curva_aprendizaje.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue=TARGET,
        palette={0: "#4472C4", 1: "#E67E22"},
        s=65,
    )
    x_min = float(data[FEATURES[0]].min()) - 1
    x_max = float(data[FEATURES[0]].max()) + 1
    if abs(weights[1]) > 1e-12:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = means[1] + standard_deviations[1] * (
            -bias / weights[1]
            - (x_line - means[0]) / standard_deviations[0] * weights[0]
        )
        plt.plot(x_line, y_line, color="black", linewidth=2, label="p = 0.5")
    else:
        vertical_x = means[0] - (bias / weights[0]) * standard_deviations[0]
        plt.axvline(vertical_x, color="black", linewidth=2, label="p = 0.5")
    misclassified = (final_probabilities >= 0.5) != target
    if np.any(misclassified):
        plt.scatter(
            features[misclassified, 0],
            features[misclassified, 1],
            facecolors="none",
            edgecolors="red",
            s=150,
            linewidths=1.5,
            label="Clasificación incorrecta",
        )
    plt.title(f"Frontera de decisión en unidades originales ({mistakes} errores)")
    plt.xlabel("Visitas al mes")
    plt.ylabel("Ticket promedio (miles de pesos)")
    plt.legend(title="Cliente recurrente / frontera")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "frontera_decision.png", dpi=160)
    plt.close()
    print(f"Gráficos guardados en: {FIGURES_DIR}")

    examples = np.array([[15.0, 250.0], [5.0, 60.0]])
    example_probabilities = predict_probabilities(
        (examples - means) / standard_deviations, weights, bias
    )
    print("\nPredicciones de ejemplo (sin cliente_id ni barrio):")
    for example, probability, expected in zip(
        examples, example_probabilities, (1, 0), strict=True
    ):
        predicted = int(probability >= 0.5)
        print(
            f"visitas={example[0]:.0f}, ticket={example[1]:.0f}: "
            f"p(recurrente)={probability:.6f}, clase={predicted}, "
            f"esperada={expected}, coincide={predicted == expected}"
        )
    return {
        "means": means,
        "standard_deviations": standard_deviations,
        "initial_weights": initial_weights,
        "initial_bias": initial_bias,
        "initial_loss": initial_loss,
        "weights": weights,
        "bias": bias,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "mistakes": mistakes,
        "examples": examples,
        "example_probabilities": example_probabilities,
        "epochs": epochs,
        "losses": losses,
        "accuracies": accuracies,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV original: {INPUT_PATH}")
    raw = pd.read_csv(INPUT_PATH)
    inspect_data(raw)
    clean, _ = clean_data(raw)
    create_eda_plots(clean)
    train_perceptron(clean)


if __name__ == "__main__":
    main()
