"""Contraste independiente del costo con aritmetica decimal y evidencia Locust."""

from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path

import pytest

from mibici_equipo14.finops import estimate_cost


REPORTS = Path(__file__).resolve().parents[1] / "reports" / "operacion"


def test_manual_cost_for_largest_scenario():
    """El resultado esperado no depende de estimate_cost."""
    window = Decimal("24") * Decimal("90") / (
        Decimal("730") * Decimal("3600")
    )
    unit = window / Decimal("8681")

    actual = estimate_cost(
        seconds=90,
        successes=8681,
        monthly_usd=24,
        planning_hours=730,
    )

    assert actual["window_usd"] == pytest.approx(
        float(window), rel=1e-12, abs=0
    )
    assert actual["per_inference_usd"] == pytest.approx(
        float(unit), rel=1e-12, abs=0
    )
    assert actual["per_1000_usd"] == pytest.approx(
        float(unit * 1000), rel=1e-12, abs=0
    )


@pytest.mark.parametrize("users", [5, 20, 50])
def test_published_summary_matches_html_and_manual_cost(users):
    """Contrasta los totales finales, no el ultimo CSV periodico."""
    summary = json.loads(
        (REPORTS / "summary.json").read_text(encoding="utf-8")
    )
    html = (REPORTS / f"locust_{users}.html").read_text(
        encoding="utf-8"
    )
    payload = html.partition("window.templateArgs = ")[2]
    assert payload, "No se encontro la evidencia final de Locust"

    report, _ = json.JSONDecoder().raw_decode(payload.lstrip())
    aggregate = next(
        item
        for item in report["requests_statistics"]
        if item["name"] == "Aggregated"
    )
    duration = (
        datetime.fromisoformat(report["end_time"])
        - datetime.fromisoformat(report["start_time"])
    ).total_seconds()
    successes = (
        int(aggregate["num_requests"])
        - int(aggregate["num_failures"])
    )

    published = next(
        item for item in summary["runs"] if item["users"] == users
    )

    assert published["requests"] == int(aggregate["num_requests"])
    assert published["failures"] == int(aggregate["num_failures"])
    assert published["successes"] == successes
    assert published["locust_seconds"] == duration

    if users == 50:
        assert successes == 8681
        assert duration == 90

    price = summary["price"]
    assert price["monthly_usd"] == 24
    assert price["planning_hours_per_month"] == 730

    window = (
        Decimal(str(price["monthly_usd"]))
        * Decimal(str(duration))
        / (
            Decimal(str(price["planning_hours_per_month"]))
            * Decimal("3600")
        )
    )
    unit = window / Decimal(successes)

    expected = {
        "window_usd": float(window),
        "per_inference_usd": float(unit),
        "per_1000_usd": float(unit * 1000),
    }
    actual = estimate_cost(
        seconds=duration,
        successes=successes,
        monthly_usd=price["monthly_usd"],
        planning_hours=price["planning_hours_per_month"],
    )

    for key, value in expected.items():
        assert actual[key] == pytest.approx(
            value, rel=1e-12, abs=0
        )
        assert published["estimated_cost"][key] == pytest.approx(
            value, rel=1e-12, abs=0
        )


@pytest.mark.parametrize(
    "monthly_usd,successes",
    [
        (48, 8681),
        (24, 8681 / 2),
    ],
)
def test_double_price_or_half_successes_doubles_unit_cost(
    monthly_usd, successes
):
    """La mitad exacta es una sensibilidad algebraica, no un conteo real."""
    base = (
        Decimal("24") * Decimal("90")
        / (Decimal("730") * Decimal("3600") * Decimal("8681"))
    )

    actual = estimate_cost(
        seconds=90,
        successes=successes,
        monthly_usd=monthly_usd,
        planning_hours=730,
    )

    assert actual["per_inference_usd"] == pytest.approx(
        float(base * 2), rel=1e-12, abs=0
    )
    assert actual["per_1000_usd"] == pytest.approx(
        float(base * 2 * 1000), rel=1e-12, abs=0
    )