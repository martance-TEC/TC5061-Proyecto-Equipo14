"""Dos entrenamientos nuevos deben conservar sus resultados con la misma semilla."""
import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.config import SETTINGS
from mibici_equipo14.modeling.advanced import build_model, resolve


@pytest.mark.parametrize('question,model,parameters', [
    ('p1', 'hgb', {'max_iter': 15}),
    ('p2', 'rf', {'n_estimators': 15, 'max_depth': 4, 'min_samples_leaf': 2}),
    ('p3', 'kmeans', {'k': 3}),
])
def test_same_seed_repeats_training(question, model, parameters):
    rng = np.random.default_rng(SETTINGS.seed)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list('abcd'))
    y = (X.a > 0).astype(int) if question == 'p2' else np.abs(X.a * 3 + X.b)
    spec = resolve(question, model, **parameters)
    first = build_model(question, spec)
    second = build_model(question, spec)
    first.fit(X.iloc[:60], y.iloc[:60])
    second.fit(X.iloc[:60], y.iloc[:60])
    np.testing.assert_allclose(first.predict(X.iloc[60:]), second.predict(X.iloc[60:]),
                               rtol=0, atol=1e-12)
    if question == 'p2':
        np.testing.assert_allclose(first.predict_proba(X.iloc[60:]),
                                   second.predict_proba(X.iloc[60:]), rtol=0, atol=1e-12)
    if question == 'p3':
        np.testing.assert_allclose(first[0].mean_, X.iloc[:60].mean(), atol=1e-12)


def test_training_keeps_the_configured_seed():
    forest = build_model('p2', resolve('p2', 'rf'))
    clusters = build_model('p3', resolve('p3', 'kmeans'))
    assert forest.random_state == SETTINGS.seed
    assert forest.n_jobs == 1
    assert clusters[-1].random_state == SETTINGS.seed
