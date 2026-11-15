"""Ejecucion del A/B de P1 con versiones registradas en MLflow."""

import os

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from .ab_testing import compare_predictions, daily_block_bootstrap, validate_ab_holdout
from .config import REGISTRY_NAMES, SETTINGS as S
from .modeling.advanced import prepare


P1_COLUMNS = ["station_id", "weekday", "hour"]


def configure_mlflow():
    """Configura Tracking y Registry desde variables de entorno."""
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    registry_uri = os.environ.get(
        "MLFLOW_REGISTRY_URI",
        tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    return {
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
    }


def validate_model_lineage(versions, client=None):
    """Valida linaje y evita entrenamiento dentro del holdout A/B."""
    client = client or MlflowClient()
    expected_data_version = None
    holdout_start = pd.Timestamp(S.val_end)
    result = {}

    for alias in ["champion", "challenger"]:
        item = versions[alias]
        run = client.get_run(item["run_id"])

        tags = {
            **dict(run.data.tags),
            **item.get("tags", {}),
        }
        params = dict(run.data.params)

        question = (
            tags.get("question")
            or tags.get("pregunta")
        )

        if str(question).lower() != "p1":
            raise ValueError(
                f"{alias} no corresponde a P1"
            )

        data_version = (
            tags.get("data_version")
            or params.get("data_version")
        )
        if not data_version:
            raise ValueError(
                f"{alias} no conserva data_version"
            )

        if expected_data_version is None:
            expected_data_version = data_version
        elif data_version != expected_data_version:
            raise ValueError(
                "Champion y challenger usan versiones de datos diferentes"
            )

        train_end = (
            params.get("train_end")
            or params.get("fit_end_exclusive")
            or tags.get("train_end")
            or tags.get("fit_end_exclusive")
        )
        if not train_end:
            raise ValueError(
                f"{alias} no conserva train_end"
            )

        train_end_timestamp = pd.Timestamp(train_end)

        if train_end_timestamp > holdout_start:
            raise ValueError(
                f"{alias} fue entrenado con datos del periodo A/B"
            )

        result[alias] = {
            "run_id": item["run_id"],
            "data_version": data_version,
            "train_end": str(train_end),
            "git_commit": tags.get("git_commit"),
        }

    result["holdout_start"] = S.val_end
    result["holdout_end"] = S.end
    result["same_data_version"] = True

    return result


def load_p1_holdout():
    """Carga el test temporal de P1: noviembre-diciembre de 2025."""
    parts, metadata, source = prepare("p1")
    holdout = parts["test"].copy()

    holdout_validation = validate_ab_holdout(holdout)

    required = set(
        P1_COLUMNS + ["departures", "time"]
    )
    missing = sorted(
        required - set(holdout.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas en el holdout P1: {missing}"
        )

    holdout = holdout.sort_values(
        ["time", "station_id"],
        kind="stable",
    ).reset_index(drop=True)

    return holdout, {
        **holdout_validation,
        "data_version": source["data_version"],
        "feature_version": metadata["feature_version"],
        "columns": P1_COLUMNS,
    }


def run_registry_ab_test(
    *,
    bootstrap_iterations=1000,
    seed=None,
    client=None,
):
    """Ejecuta champion vs challenger sobre el mismo holdout P1."""
    configure_mlflow()
    client = client or MlflowClient()

    versions = resolve_p1_versions(client)
    lineage = validate_model_lineage(
        versions,
        client,
    )
    holdout, holdout_metadata = load_p1_holdout()

    inputs = holdout[P1_COLUMNS].astype({
        "station_id": str,
        "weekday": "int32",
        "hour": "int32",
    })
    target = holdout[
        "departures"
    ].to_numpy(dtype=float)

    champion_model = mlflow.pyfunc.load_model(
        versions["champion"]["model_uri"]
    )
    challenger_model = mlflow.pyfunc.load_model(
        versions["challenger"]["model_uri"]
    )

    champion_prediction = np.asarray(
        champion_model.predict(inputs),
        dtype=float,
    )
    challenger_prediction = np.asarray(
        challenger_model.predict(inputs),
        dtype=float,
    )

    comparison = compare_predictions(
        target,
        champion_prediction,
        challenger_prediction,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    daily_bootstrap = daily_block_bootstrap(
        holdout["time"],
        target,
        champion_prediction,
        challenger_prediction,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    return {
        "experiment": "offline_paired_ab_test",
        "model_name": versions["name"],
        "versions": versions,
        "lineage": lineage,
        "holdout": holdout_metadata,
        "comparison": comparison,
        "daily_block_bootstrap": daily_bootstrap,
    }


def resolve_p1_versions(client=None):
    """Resuelve champion y challenger y valida versiones distintas y READY."""
    client = client or MlflowClient()
    name = REGISTRY_NAMES["p1"]

    try:
        champion = client.get_model_version_by_alias(
            name,
            "champion",
        )
    except Exception as error:
        raise ValueError(
            "Champion P1 no disponible en MLflow Registry"
        ) from error

    try:
        challenger = client.get_model_version_by_alias(
            name,
            "challenger",
        )
    except Exception as error:
        raise ValueError(
            "Challenger P1 no disponible en MLflow Registry"
        ) from error

    if str(champion.version) == str(challenger.version):
        raise ValueError(
            "Champion y challenger deben ser versiones distintas"
        )

    for alias, version in [
        ("champion", champion),
        ("challenger", challenger),
    ]:
        if version.status != "READY":
            raise ValueError(
                f"La version {alias} de P1 no esta READY"
            )

    return {
        "name": name,
        "champion": {
            "alias": "champion",
            "version": str(champion.version),
            "run_id": champion.run_id,
            "model_uri": f"models:/{name}@champion",
            "tags": dict(champion.tags),
        },
        "challenger": {
            "alias": "challenger",
            "version": str(challenger.version),
            "run_id": challenger.run_id,
            "model_uri": f"models:/{name}@challenger",
            "tags": dict(challenger.tags),
        },
    }
