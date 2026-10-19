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
