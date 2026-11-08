"""API académica para el promedio histórico de demanda P1"""
import argparse
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI
from mlflow import MlflowClient
from mlflow.types.schema import ColSpec, Schema
from pydantic import BaseModel, ConfigDict, Field

from .config import REGISTRY_NAMES


class PredictionInstance(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    station_id: str = Field(min_length=1)
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    instances: list[PredictionInstance] = Field(min_length=1)


def create_app(
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    model_name: str = REGISTRY_NAMES['p1'],
    model_version: int | None = None,
    model_alias: str | None = None,
) -> FastAPI:
    if model_version is not None and model_alias is not None:
        raise ValueError('Usa una versión o un alias, no ambos')

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri is not None:
            mlflow.set_registry_uri(registry_uri)

        version = model_version
        if version is None:
            version = MlflowClient().get_model_version_by_alias(
                model_name, model_alias or 'champion',
            ).version
        version = str(int(version))
        model = mlflow.pyfunc.load_model(f'models:/{model_name}/{version}')
        expected = Schema([
            ColSpec('string', 'station_id'),
            ColSpec('integer', 'weekday'),
            ColSpec('integer', 'hour'),
        ])
        if model.metadata.get_input_schema() != expected:
            raise RuntimeError(
                'La firma P1 requiere station_id:string, weekday:int32 y hour:int32',
            )
        app.state.model = model
        app.state.model_version = version
        yield

    app = FastAPI(title='MiBici P1', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, str]:
        return {
            'status': 'ready',
            'model_name': model_name,
            'model_version': app.state.model_version,
        }

    @app.post('/predict')
    def predict(payload: PredictionRequest) -> dict[str, list[float]]:
        frame = pd.DataFrame([row.model_dump() for row in payload.instances])
        # Tipos de entrada de MLflow
        frame = frame.astype({'weekday': 'int32', 'hour': 'int32'})
        return {'predictions': app.state.model.predict(frame).tolist()}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--tracking-uri')
    parser.add_argument('--registry-uri')
    parser.add_argument('--model-name', default=REGISTRY_NAMES['p1'])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--model-version', type=int)
    selection.add_argument('--model-alias')
    args = parser.parse_args()
    app = create_app(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        model_name=args.model_name,
        model_version=args.model_version,
        model_alias=args.model_alias,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
