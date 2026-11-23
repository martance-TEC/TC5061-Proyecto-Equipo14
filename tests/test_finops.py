"""Unidades y denominador del costo por inferencia."""
import pytest
from mibici_equipo14.finops import estimate_cost, memory_mib, cpu_seconds, final_statistics


def test_cost_uses_successes_and_full_instance_time():
    value = estimate_cost(3600, 1000, monthly_usd=730)
    assert value == {'window_usd': 1., 'per_inference_usd': .001, 'per_1000_usd': 1.}
    assert estimate_cost(3600, 500, monthly_usd=730)['per_1000_usd'] == 2.


@pytest.mark.parametrize('seconds,successes', [(0, 1), (1, 0), (-1, 5)])
def test_empty_or_invalid_measurements_are_not_zero_cost(seconds, successes):
    with pytest.raises(ValueError):
        estimate_cost(seconds, successes)


def test_resource_units_are_comparable():
    assert memory_mib('1GiB / 2GiB') == 1024
    assert memory_mib('104.8576MB / 805.3MB') == pytest.approx(100)
    assert cpu_seconds('1m2.5s') == 62.5


def test_final_totals_come_from_locust_report_without_executing_script():
    html = '<script>window.templateArgs = {"requests_statistics":[{"name":"/predict","num_requests":12},{"name":"Aggregated","num_requests":13}]}\nwindow.theme="light"</script>'
    assert final_statistics(html)['num_requests'] == 13
