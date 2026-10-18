from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    results = json.loads((folder / "reproduccion_milton.json").read_text())["results"]["P1"]["models"]
    names = ["seasonal_naive", "station_weekday_hour_mean", "hist_gradient_boosting"]
    labels = ["Semana anterior", "Promedio histórico", "Boosting Poisson"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    positions = np.arange(len(names))
    width = .34
    for shift, split, color in [(-width / 2, "val", "#777777"), (width / 2, "test", "#243e54")]:
        values = [results[name][split + "_mae"] for name in names]
        bars = ax.bar(positions + shift, values, width,
                      label={"val": "Validación", "test": "Prueba"}[split], color=color)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("MAE · viajes por estación y hora")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    destination = folder / "figures" / "p1_milton_comparacion.png"
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
