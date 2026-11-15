"""Pruebas del motor A/B para las versiones de P1."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.ab_testing import (
    bootstrap_delta_mae,
    compare_predictions,
    regression_metrics,
    validate_ab_holdout,
    validate_predictions,
)


def test_regression_metrics_have_known_values():
    target = [1, 2, 4, 3]
    prediction = [1, 3, 2, 3]

    result = regression_metrics(target, prediction)

    assert result["mae"] == pytest.approx(0.75)
    assert result["rmse"] == pytest.approx(np.sqrt(1.25))
    assert result["wape"] == pytest.approx(0.30)


def test_paired_comparison_counts_wins_ties_and_metrics():
    target = [1, 2, 4, 3]
    champion = [1, 3, 2, 3]
    challenger = [1, 2, 3, 4]

    result = compare_predictions(
        target,
        champion,
        challenger,
        bootstrap_iterations=500,
        seed=42,
    )

    assert result["rows"] == 4
    assert result["champion"]["mae"] == pytest.approx(0.75)
    assert result["challenger"]["mae"] == pytest.approx(0.50)
    assert result["paired"]["champion_wins"] == 1
    assert result["paired"]["challenger_wins"] == 2
    assert result["paired"]["ties"] == 1
    assert result["paired"]["mean_error_difference"] == pytest.approx(-0.25)
    assert result["bootstrap"]["delta_mae"] == pytest.approx(-0.25)


def test_bootstrap_is_reproducible_with_the_same_seed():
    champion_error = [0, 1, 2, 0]
    challenger_error = [0, 0, 1, 1]

    first = bootstrap_delta_mae(
        champion_error,
        challenger_error,
        iterations=1000,
        seed=42,
    )
    second = bootstrap_delta_mae(
        champion_error,
        challenger_error,
        iterations=1000,
        seed=42,
    )

    assert first == second
    assert first["ci95_lower"] <= first["delta_mae"]
    assert first["delta_mae"] <= first["ci95_upper"]


def test_ab_result_is_json_serializable_without_nan():
    result = compare_predictions(
        [1, 2, 4, 3],
        [1, 3, 2, 3],
        [1, 2, 3, 4],
        bootstrap_iterations=100,
    )

    serialized = json.dumps(result, allow_nan=False)

    assert "NaN" not in serialized
    assert "Infinity" not in serialized


def test_ab_holdout_accepts_only_november_and_december():
    valid = pd.DataFrame({
        "time": [
            "2025-11-01 00:00:00",
            "2025-12-31 22:00:00",
        ],
    })

    result = validate_ab_holdout(valid)

    assert result == {
        "start": "2025-11-01",
        "end": "2026-01-01",
        "rows": 2,
    }


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2025-10-31 23:00:00"],
        ["2026-01-01 00:00:00"],
        ["fecha-invalida"],
    ],
)
def test_ab_holdout_rejects_leakage_and_invalid_dates(timestamps):
    with pytest.raises(ValueError):
        validate_ab_holdout(pd.DataFrame({"time": timestamps}))


@pytest.mark.parametrize(
    "target,champion,challenger",
    [
        ([], [], []),
        ([1, 2], [1], [1, 2]),
        ([1, np.nan], [1, 2], [1, 2]),
        ([1, 2], [1, np.inf], [1, 2]),
    ],
)
def test_invalid_prediction_vectors_are_rejected(
    target,
    champion,
    challenger,
):
    with pytest.raises(ValueError):
        validate_predictions(target, champion, challenger)


def test_wape_rejects_zero_target_volume():
    with pytest.raises(ValueError, match="WAPE"):
        regression_metrics([0, 0], [0, 1])


def test_bootstrap_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        bootstrap_delta_mae([1, 2], [1], iterations=100)

    with pytest.raises(ValueError):
        bootstrap_delta_mae([1, 2], [1, 2], iterations=0)


def test_daily_block_bootstrap_favors_challenger_across_days():
    from mibici_equipo14.ab_testing import daily_block_bootstrap

    dates = [
        "2025-11-01 08:00:00",
        "2025-11-01 09:00:00",
        "2025-11-02 08:00:00",
        "2025-11-02 09:00:00",
        "2025-11-03 08:00:00",
        "2025-11-03 09:00:00",
    ]
    target = [10, 10, 10, 10, 10, 10]
    champion = [12, 12, 13, 13, 11, 11]
    challenger = [11, 11, 11, 11, 10, 10]

    result = daily_block_bootstrap(
        dates,
        target,
        champion,
        challenger,
        iterations=1000,
        seed=42,
    )

    assert result["days"] == 3
    assert result["iterations"] == 1000
    assert result["seed"] == 42
    assert result["challenger_better_days"] == 3
    assert result["champion_better_days"] == 0
    assert result["tied_days"] == 0
    assert result["delta_mae"] < 0
    assert result["ci95_upper"] < 0
    assert result["statistical_result"] == "challenger"


def test_daily_block_bootstrap_is_reproducible():
    from mibici_equipo14.ab_testing import daily_block_bootstrap

    dates = [
        "2025-11-01",
        "2025-11-01",
        "2025-11-02",
        "2025-11-02",
    ]
    target = [4, 6, 3, 5]
    champion = [5, 8, 4, 7]
    challenger = [4, 7, 3, 6]

    first = daily_block_bootstrap(
        dates,
        target,
        champion,
        challenger,
        iterations=500,
        seed=42,
    )
    second = daily_block_bootstrap(
        dates,
        target,
        champion,
        challenger,
        iterations=500,
        seed=42,
    )

    assert first == second


@pytest.mark.parametrize(
    "dates,target,champion,challenger,iterations",
    [
        (["2025-11-01"], [1, 2], [1, 2], [1, 2], 100),
        (["fecha-invalida"], [1], [1], [1], 100),
        (["2025-11-01"], [1], [1], [1], 0),
    ],
)
def test_daily_block_bootstrap_rejects_invalid_inputs(
    dates,
    target,
    champion,
    challenger,
    iterations,
):
    from mibici_equipo14.ab_testing import daily_block_bootstrap

    with pytest.raises(ValueError):
        daily_block_bootstrap(
            dates,
            target,
            champion,
            challenger,
            iterations=iterations,
            seed=42,
        )
