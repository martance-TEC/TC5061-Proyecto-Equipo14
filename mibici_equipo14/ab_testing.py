"""Comparacion A/B reproducible para dos versiones del modelo P1."""

import numpy as np
import pandas as pd

from .config import SETTINGS as S


def _finite_vector(values, name):
    """Convierte valores a un vector finito, no vacio y unidimensional."""
    vector = np.asarray(values, dtype=float)

    if vector.ndim != 1:
        raise ValueError(f"{name} debe ser un vector unidimensional")
    if vector.size == 0:
        raise ValueError(f"{name} no puede estar vacio")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contiene NaN o infinito")

    return vector


def validate_predictions(y_true, champion, challenger):
    """Valida que target y predicciones sean comparables fila por fila."""
    y = _finite_vector(y_true, "y_true")
    prediction_champion = _finite_vector(champion, "champion")
    prediction_challenger = _finite_vector(challenger, "challenger")

    if not (
        len(y) == len(prediction_champion) == len(prediction_challenger)
    ):
        raise ValueError(
            "Target, champion y challenger deben tener el mismo numero de filas"
        )

    return y, prediction_champion, prediction_challenger


def regression_metrics(y_true, prediction):
    """Calcula MAE, RMSE y WAPE para una version del modelo."""
    y = _finite_vector(y_true, "y_true")
    predicted = _finite_vector(prediction, "prediction")

    if len(y) != len(predicted):
        raise ValueError("Target y prediccion deben tener el mismo numero de filas")

    absolute_error = np.abs(y - predicted)
    denominator = np.abs(y).sum()

    if denominator == 0:
        raise ValueError("WAPE no esta definido cuando el target suma cero")

    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean((y - predicted) ** 2))),
        "wape": float(absolute_error.sum() / denominator),
    }


def bootstrap_delta_mae(
    champion_absolute_error,
    challenger_absolute_error,
    *,
    iterations=1000,
    seed=None,
):
    """Estima IC 95% de delta MAE: challenger menos champion."""
    champion_error = _finite_vector(
        champion_absolute_error,
        "champion_absolute_error",
    )
    challenger_error = _finite_vector(
        challenger_absolute_error,
        "challenger_absolute_error",
    )

    if len(champion_error) != len(challenger_error):
        raise ValueError("Los errores pareados deben tener el mismo tamaño")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations debe ser un entero positivo")

    paired_delta = challenger_error - champion_error
    generator = np.random.default_rng(S.seed if seed is None else seed)
    bootstrap_means = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        indices = generator.integers(
            0,
            len(paired_delta),
            size=len(paired_delta),
        )
        bootstrap_means[iteration] = paired_delta[indices].mean()
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        "iterations": iterations,
        "seed": S.seed if seed is None else seed,
        "delta_mae": float(paired_delta.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def compare_predictions(
    y_true,
    champion,
    challenger,
    *,
    bootstrap_iterations=1000,
    seed=None,
    tie_tolerance=1e-12,
):
    """Compara dos versiones sobre exactamente las mismas observaciones."""
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance no puede ser negativo")

    y, prediction_champion, prediction_challenger = validate_predictions(
        y_true,
        champion,
        challenger,
    )

    champion_error = np.abs(y - prediction_champion)
    challenger_error = np.abs(y - prediction_challenger)
    delta = challenger_error - champion_error

    challenger_wins = int((delta < -tie_tolerance).sum())
    champion_wins = int((delta > tie_tolerance).sum())
    ties = int((np.abs(delta) <= tie_tolerance).sum())
    rows = len(y)

    bootstrap = bootstrap_delta_mae(
        champion_error,
        challenger_error,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    if bootstrap["ci95_upper"] < 0:
        statistical_result = "challenger"
    elif bootstrap["ci95_lower"] > 0:
        statistical_result = "champion"
    else:
        statistical_result = "inconclusive"

    return {
        "rows": rows,
        "champion": regression_metrics(y, prediction_champion),
        "challenger": regression_metrics(y, prediction_challenger),
        "paired": {
            "champion_wins": champion_wins,
            "challenger_wins": challenger_wins,
            "ties": ties,
            "champion_win_rate": float(champion_wins / rows),
            "challenger_win_rate": float(challenger_wins / rows),
            "tie_rate": float(ties / rows),
            "mean_error_difference": float(delta.mean()),
        },
        "bootstrap": bootstrap,
        "statistical_result": statistical_result,
    }


def validate_ab_holdout(frame, time_column="time"):
    """Verifica que el A/B utilice solo noviembre-diciembre de 2025."""
    if time_column not in frame:
        raise ValueError(f"Falta la columna temporal {time_column}")
    if frame.empty:
        raise ValueError("El conjunto A/B no puede estar vacio")

    timestamps = pd.to_datetime(frame[time_column], errors="coerce")

    if timestamps.isna().any():
        raise ValueError("El conjunto A/B contiene fechas invalidas")

    start = pd.Timestamp(S.val_end)
    end = pd.Timestamp(S.end)
    outside = (timestamps < start) | (timestamps >= end)

    if outside.any():
        raise ValueError(
            f"El A/B debe usar solo el periodo [{S.val_end}, {S.end})"
        )

    return {
        "start": S.val_end,
        "end": S.end,
        "rows": int(len(frame)),
    }


def daily_block_bootstrap(
    dates,
    y_true,
    champion,
    challenger,
    *,
    iterations=1000,
    seed=None,
):
    """Bootstrap pareado por dias completos para respetar dependencia temporal."""
    y, prediction_champion, prediction_challenger = validate_predictions(
        y_true,
        champion,
        challenger,
    )

    dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    if len(dates) != len(y):
        raise ValueError(
            "Fechas, target y predicciones deben tener el mismo numero de filas"
        )
    if dates.isna().any():
        raise ValueError("Las fechas contienen valores invalidos")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations debe ser un entero positivo")

    errors = pd.DataFrame({
        "date": dates.dt.date,
        "champion_error": np.abs(y - prediction_champion),
        "challenger_error": np.abs(y - prediction_challenger),
    })

    daily = errors.groupby("date", sort=True).agg(
        champion_mae=("champion_error", "mean"),
        challenger_mae=("challenger_error", "mean"),
    )

    daily["delta_mae"] = (
        daily["challenger_mae"]
        - daily["champion_mae"]
    )

    delta = daily["delta_mae"].to_numpy(dtype=float)
    generator = np.random.default_rng(
        S.seed if seed is None else seed
    )
    bootstrap = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sample = generator.choice(
            delta,
            size=len(delta),
            replace=True,
        )
        bootstrap[iteration] = sample.mean()

    lower, upper = np.percentile(
        bootstrap,
        [2.5, 97.5],
    )

    if upper < 0:
        statistical_result = "challenger"
    elif lower > 0:
        statistical_result = "champion"
    else:
        statistical_result = "inconclusive"

    return {
        "method": "daily_block_bootstrap",
        "days": int(len(delta)),
        "iterations": iterations,
        "seed": S.seed if seed is None else seed,
        "delta_mae": float(delta.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "challenger_better_days": int((delta < 0).sum()),
        "champion_better_days": int((delta > 0).sum()),
        "tied_days": int((delta == 0).sum()),
        "statistical_result": statistical_result,
    }
