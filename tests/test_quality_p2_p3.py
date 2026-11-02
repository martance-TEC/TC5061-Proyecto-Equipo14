import pandas as pd
import numpy as np

from mibici_equipo14.config import SETTINGS as S
from mibici_equipo14.features import station_profiles
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
    """Genera el perfil de una estación con cuatro semanas de actividad."""
    times = pd.date_range(S.start, periods=28 * 24, freq="h")
    hourly = pd.DataFrame({
        "station_id": "station_100",
        "time": times,
        "departures": 1,
        "arrivals": 1,
    })
    end = times[-1] + pd.Timedelta(hours=1)
    return station_profiles(hourly, times[0], end)


def test_p2_valid_data_is_accepted():
    frame = valid_p2_frame()

    issues = table_issues(frame, "p2", "train")

    assert issues == []


def test_p2_invalid_risk_is_rejected():
    frame = valid_p2_frame()
    frame.loc[0, "risk"] = 2

    issues = table_issues(frame, "p2", "train")

    assert "risk debe contener solamente 0 y 1" in issues


def test_p3_valid_normalized_profile_is_accepted():
    frame = valid_p3_frame()
    columns = [f"{kind}_{hour}" for kind in ("departures", "arrivals")
               for hour in range(48)]
    proportions = frame[columns]

    assert frame.shape == (1, 99)
    assert proportions.ge(0).all().all()
    assert proportions.le(1).all().all()
    np.testing.assert_allclose(proportions.sum(axis=1), [1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame["weekend_share"], [2 / 7], rtol=0, atol=1e-12)
    assert table_issues(frame, "p3", "train") == []


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
