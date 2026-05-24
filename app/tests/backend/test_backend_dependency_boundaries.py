from __future__ import annotations

from pathlib import Path


def _iter_python_files(root: str):
    yield from Path(root).rglob('*.py')


def test_core_has_no_ml_imports() -> None:
    forbidden = [
        'from ml_service',
        'import ml_service',
        'import torch',
        'from torch',
        'import keras',
        'from keras',
        'import sklearn',
        'from sklearn',
    ]
    for path in _iter_python_files('app/server/core_service'):
        text = path.read_text(encoding='utf-8')
        hits = [item for item in forbidden if item in text]
        assert not hits, f'{path}: forbidden imports {hits}'


def test_shared_has_no_service_imports() -> None:
    forbidden = ['from core_service', 'import core_service', 'from ml_service', 'import ml_service']
    for path in _iter_python_files('app/server/shared'):
        text = path.read_text(encoding='utf-8')
        hits = [item for item in forbidden if item in text]
        assert not hits, f'{path}: forbidden imports {hits}'
