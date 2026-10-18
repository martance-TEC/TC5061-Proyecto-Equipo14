"""Aislamiento temporal y geográfico de los perfiles de P3."""
import pandas as pd
from pandas.testing import assert_frame_equal
from mibici_equipo14.features import station_profiles

def example():
    return pd.DataFrame({"station_id": ["2"]*96,
        "time": pd.date_range("2025-08-30", periods=96, freq="h"),
        "departures": [1]*96, "arrivals": [2]*96})

def test_future_activity_cannot_change_training_profiles():
    frame = example()
    before = station_profiles(frame, "2025-08-30", "2025-09-01")
    frame.loc[frame.time >= "2025-09-01", ["departures", "arrivals"]] = 99999
    after = station_profiles(frame, "2025-08-30", "2025-09-01")
    assert_frame_equal(before, after)

def test_station_coordinates_cannot_change_usage_profiles():
    frame = example()
    before = station_profiles(frame, "2025-08-30", "2025-09-01")
    frame["latitude"] = 0.; frame["longitude"] = 0.
    frame.loc[0, "latitude"] = 90.
    after = station_profiles(frame, "2025-08-30", "2025-09-01")
    assert_frame_equal(before, after)
    assert before.shape[1] == 99
