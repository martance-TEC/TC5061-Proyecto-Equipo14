"""Configuración central compartida; intervalos temporales semiabiertos."""
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[1]
    seed: int = 42
    start: str = "2025-01-01"
    train_end: str = "2025-09-01"
    val_end: str = "2025-11-01"
    end: str = "2026-01-01"
    experiment: str = "mibici-equipo14"
    max_iter: int = 100
    chunk_size: int = 200000
    initial_fraction: float = 0.5
    low_fraction: float = 0.1
    high_fraction: float = 0.9
    horizon_hours: int = 3
SETTINGS = Settings()

# La campaña se decide antes de evaluar. No se cambian fechas ni features.
REGISTRY_NAMES = {
    'p1': 'mibici-equipo14-p1-demanda',
    'p2': 'mibici-equipo14-p2-riesgo',
    'p3': 'mibici-equipo14-p3-grupos',
}
CAMPAIGN = {
    'p1': [
        dict(model='seasonal_naive', role='baseline'),
        dict(model='historical_mean', role='previous_champion'),
        dict(model='hgb', max_iter=150, learning_rate=.1, max_leaf_nodes=15),
        dict(model='hgb', max_iter=250, learning_rate=.05, max_leaf_nodes=31),
        dict(model='hgb', max_iter=350, learning_rate=.03, max_leaf_nodes=31),
    ],
    'p2': [
        dict(model='dummy', role='baseline'),
        dict(model='rf', role='previous_champion', n_estimators=200, max_depth=12, min_samples_leaf=10),
        dict(model='rf', n_estimators=100, max_depth=8, min_samples_leaf=20),
        dict(model='rf', n_estimators=300, max_depth=12, min_samples_leaf=10),
        dict(model='rf', n_estimators=300, max_depth=16, min_samples_leaf=20),
    ],
    'p3': [dict(model='kmeans', k=k, role='baseline' if k==2 else 'previous_champion' if k==3 else 'candidate') for k in range(2,7)],
}
DEFAULT_MODELS = {'p1':'hgb', 'p2':'rf', 'p3':'kmeans'}
MODEL_DEFAULTS = {
    'hgb': dict(max_iter=100, learning_rate=.1, max_leaf_nodes=31),
    'rf': dict(n_estimators=200, max_depth=12, min_samples_leaf=10),
    'kmeans': dict(k=3), 'seasonal_naive': {}, 'historical_mean': {}, 'dummy': {},
}
REPORT_DIR = SETTINGS.root / 'reports' / 'segunda_entrega'
