from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _iter_ts_like_files(root: Path):
    for pattern in ('*.ts', '*.tsx'):
        yield from root.rglob(pattern)


def test_core_frontend_has_no_ml_imports_or_endpoints() -> None:
    root = Path('app/client/src')
    forbidden_tokens = [
        'MachineLearningPage',
        '/api/training',
        '/training/',
        'services/training',
        'services/datasetBuilder',
        'from \'./datasetBuilder\'',
        'from "./datasetBuilder"',
        'from \'./training\'',
        'from "./training"',
    ]

    violations: list[str] = []
    for path in _iter_ts_like_files(root):
        text = _read_text(path)
        for token in forbidden_tokens:
            if token in text:
                violations.append(f'{path}: {token}')

    assert not violations, '\n'.join(violations)


def test_ml_frontend_has_no_core_page_imports() -> None:
    root = Path('app/ml_client/src')
    forbidden_tokens = [
        'ConfigPage',
        'ModelsPage',
        'adsorptionModels',
        'startFittingJob',
        'pollFittingJobUntilComplete',
        'fetchNistDataForFitting',
        'fetchDatasetByName',
        'fetchDatasetNames',
        'loadDataset(',
        'services/fitting',
        'services/datasets',
        'services/nist',
    ]

    violations: list[str] = []
    for path in _iter_ts_like_files(root):
        text = _read_text(path)
        for token in forbidden_tokens:
            if token in text:
                violations.append(f'{path}: {token}')

    assert not violations, '\n'.join(violations)


def test_core_proxy_has_no_ml_routes() -> None:
    text = Path('app/client/proxy.conf.cjs').read_text(encoding='utf-8')
    assert '/api/training' not in text


def test_ml_proxy_has_only_ml_routes() -> None:
    text = Path('app/ml_client/proxy.conf.cjs').read_text(encoding='utf-8')
    assert '/api/training' in text
