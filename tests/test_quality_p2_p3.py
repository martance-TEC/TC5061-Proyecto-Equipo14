"""Pruebas de calidad específicas para P2 y P3."""

import pandas as pd

from mibici_equipo14.config import SETTINGS as S
from mibici_equipo14.modeling.advanced import P2_COLUMNS
from mibici_equipo14.quality_data import table_issues


def valid_p2_frame():
    row = {column: 1.0 for column in P2_COLUMNS}

    row.update(
        {
            "station_id": "station_100",
            "time": pd.Timestamp(S.start) + pd.Timedelta(days=1),
            "departures": 1,
            "arrivals": 1,
            "capacity": 20,
            "risk": 0,
        }
    )

    return pd.DataFrame([row])


def valid_p3_frame():
    row = {}

    for hour in range(48):
        row[f"departures_{hour}"] = 1.0
        row[f"arrivals_{hour}"] = 1.0

    row["log_volume"] = 1.0
    row["weekend_share"] = 0.25
    row["balance"] = 0.0

    return pd.DataFrame(
        [row],
        index=["station_100"],
    )


def test_p2_valid_data_is_accepted():
    frame = valid_p2_frame()

    issues = table_issues(frame, "p2", "train")

    assert issues == []


def test_p2_invalid_risk_is_rejected():
    frame = valid_p2_frame()
    frame.loc[0, "risk"] = 2

    issues = table_issues(frame, "p2", "train")

    assert "risk debe contener solamente 0 y 1" in issues


def test_p3_duplicate_station_is_rejected():
    frame = valid_p3_frame()
    duplicate = pd.concat([frame, frame])

    issues = table_issues(duplicate, "p3", "train")

    assert "El índice de estaciones debe ser único y no nulo" in issues


def test_p3_invalid_weekend_share_is_rejected():
    frame = valid_p3_frame()
    frame.loc["station_100", "weekend_share"] = 1.2

    issues = table_issues(frame, "p3", "train")

    assert "weekend_share debe estar entre 0 y 1" in issues