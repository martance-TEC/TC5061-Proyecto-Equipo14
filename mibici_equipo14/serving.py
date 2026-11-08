"""API académica para el promedio histórico de demanda P1"""
import argparse
from typing import Annotated, Literal
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.types.schema import ColSpec, Schema
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import REGISTRY_NAMES


class PredictionInstance(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    station_id: str = Field(min_length=1)
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)

    @field_validator('station_id')
    @classmethod
    def station_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('station_id no puede contener solo espacios')
        return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    instances: list[PredictionInstance] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predictions: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=1)


class HealthResponse(BaseModel):
    status: Literal['ready']
    model_name: str
    model_version: str


def create_app(
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    model_name: str = REGISTRY_NAMES['p1'],
    model_version: int | None = None,
    model_alias: str | None = None,
) -> FastAPI:
    if model_version is not None and model_version < 1:
        raise ValueError('La versión del modelo debe ser positiva')
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

    @app.get('/health', response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {
            'status': 'ready',
            'model_name': model_name,
            'model_version': app.state.model_version,
        }

    @app.post('/predict', response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        frame = pd.DataFrame([row.model_dump() for row in payload.instances])
        # Tipos de entrada de MLflow
        frame = frame.astype({'weekday': 'int32', 'hour': 'int32'})
        values = app.state.model.predict(frame).tolist()
        if len(values) != len(payload.instances):
            raise HTTPException(status_code=500, detail='Cantidad de predicciones inválida')
        try:
            return PredictionResponse(predictions=values)
        except ValidationError as error:
            raise HTTPException(status_code=500, detail='Salida del modelo inválida') from error

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
