"""Modelos cargables que conservan las reglas usadas en la evaluación."""
import numpy as np
import pandas as pd
import mlflow.pyfunc
from sklearn.base import BaseEstimator, RegressorMixin


class SeasonalNaive(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X['departures_lag168'].to_numpy(dtype=float)


class HistoricalMean(RegressorMixin, BaseEstimator):
    keys = ['station_id', 'weekday', 'hour']

    def fit(self, X, y):
        frame = X[self.keys].copy()
        frame['target'] = np.asarray(y)
        self.means_ = frame.groupby(self.keys).target.mean()
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        index = pd.MultiIndex.from_frame(X[self.keys])
        return self.means_.reindex(index).fillna(self.fallback_).to_numpy()


class RiskDecision(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, threshold, columns):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.columns = list(columns)

    def predict(self, context, model_input, params=None):
        probability = self.estimator.predict_proba(model_input[self.columns])[
            :, list(self.estimator.classes_).index(1)]
        return pd.DataFrame({
            'risk_probability': probability,
            'risk_class': (probability >= self.threshold).astype('int64'),
        }, index=model_input.index)
