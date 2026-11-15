"""Pruebas del conector A/B con MLflow Registry."""

from types import SimpleNamespace

import pytest

from mibici_equipo14.ab_registry import (
    resolve_p1_versions,
    validate_model_lineage,
)


class FakeClient:
    def __init__(
        self,
        *,
        same_version=False,
        missing_challenger=False,
        challenger_train_end="2025-11-01",
        different_data_version=False,
        challenger_status="READY",
    ):
        self.same_version = same_version
        self.missing_challenger = missing_challenger
        self.challenger_train_end = challenger_train_end
        self.different_data_version = different_data_version
        self.challenger_status = challenger_status

    def get_model_version_by_alias(self, name, alias):
        assert name == "mibici-equipo14-p1-demanda"

        if alias == "challenger" and self.missing_challenger:
            raise RuntimeError("missing challenger")

        version = "1" if alias == "champion" else (
            "1" if self.same_version else "2"
        )

        run_id = (
            "run-champion"
            if alias == "champion"
            else "run-challenger"
        )

        status = (
            "READY"
            if alias == "champion"
            else self.challenger_status
        )

        data_version = (
            "different"
            if alias == "challenger"
            and self.different_data_version
            else "data-v1"
        )

        question_tags = (
            {"question": "p1"}
            if alias == "champion"
            else {"pregunta": "P1"}
        )

        return SimpleNamespace(
            version=version,
            run_id=run_id,
            status=status,
            tags={
                **question_tags,
                "data_version": data_version,
            },
        )

    def get_run(self, run_id):
        is_champion = run_id == "run-champion"

        train_end = (
            "2025-09-01"
            if is_champion
            else self.challenger_train_end
        )

        data_version = (
            "different"
            if not is_champion
            and self.different_data_version
            else "data-v1"
        )

        tags = {
            "git_commit": "abc123",
            "data_version": data_version,
        }

        if is_champion:
            tags["question"] = "p1"
            params = {
                "train_end": train_end,
                "data_version": data_version,
            }
        else:
            tags["pregunta"] = "P1"
            params = {
                "fit_end_exclusive": train_end,
                "original_train_end": "2025-09-01",
                "data_version": data_version,
            }

        return SimpleNamespace(
            data=SimpleNamespace(
                tags=tags,
                params=params,
            ),
        )


def test_registry_resolves_distinct_ready_versions():
    result = resolve_p1_versions(FakeClient())

    assert result["name"] == "mibici-equipo14-p1-demanda"
    assert result["champion"]["version"] == "1"
    assert result["challenger"]["version"] == "2"
    assert result["champion"]["model_uri"].endswith("@champion")
    assert result["challenger"]["model_uri"].endswith("@challenger")


def test_registry_rejects_missing_challenger():
    with pytest.raises(ValueError, match="Challenger"):
        resolve_p1_versions(
            FakeClient(missing_challenger=True)
        )


def test_registry_rejects_same_version_for_both_aliases():
    with pytest.raises(ValueError, match="distintas"):
        resolve_p1_versions(
            FakeClient(same_version=True)
        )


def test_registry_rejects_non_ready_challenger():
    with pytest.raises(ValueError, match="READY"):
        resolve_p1_versions(
            FakeClient(
                challenger_status="PENDING_REGISTRATION"
            )
        )


def test_lineage_accepts_champion_and_milton_challenger_contracts():
    client = FakeClient()
    versions = resolve_p1_versions(client)

    result = validate_model_lineage(
        versions,
        client,
    )

    assert result["champion"]["train_end"] == "2025-09-01"
    assert result["challenger"]["train_end"] == "2025-11-01"
    assert result["holdout_start"] == "2025-11-01"
    assert result["holdout_end"] == "2026-01-01"
    assert result["same_data_version"] is True


def test_lineage_rejects_training_inside_holdout():
    client = FakeClient(
        challenger_train_end="2025-12-01"
    )
    versions = resolve_p1_versions(client)

    with pytest.raises(ValueError, match="periodo A/B"):
        validate_model_lineage(
            versions,
            client,
        )


def test_lineage_rejects_different_data_versions():
    client = FakeClient(
        different_data_version=True
    )
    versions = resolve_p1_versions(client)

    with pytest.raises(
        ValueError,
        match="versiones de datos diferentes",
    ):
        validate_model_lineage(
            versions,
            client,
        )
