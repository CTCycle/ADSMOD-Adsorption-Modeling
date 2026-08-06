from __future__ import annotations

from pathlib import Path

###############################################################################
def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

###############################################################################
def _iter_ts_like_files(root: Path):
    for pattern in ('*.ts', '*.tsx'):
        yield from root.rglob(pattern)

###############################################################################
def test_unified_frontend_keeps_expected_training_routes() -> None:
    text = Path('app/client/src/app/app.routes.ts').read_text(encoding='utf-8')
    assert "path: 'training'" in text
    assert "path: 'training/:view'" in text
    assert 'MachineLearningPageComponent' in text

###############################################################################
def test_unified_frontend_keeps_core_and_training_pages() -> None:
    root = Path('app/client/src')
    required_tokens = [
        'CustomDatasetsPageComponent',
        'ModelsPageComponent',
        'MachineLearningPageComponent',
    ]

    found = set()
    for path in _iter_ts_like_files(root):
        text = _read_text(path)
        for token in required_tokens:
            if token in text:
                found.add(token)

    assert found == set(required_tokens)

###############################################################################
def test_unified_proxy_routes_training_before_core_api() -> None:
    text = Path('app/client/proxy.conf.cjs').read_text(encoding='utf-8')
    assert '/api/training' in text
    assert "'/api'" in text
    assert text.index("'/api/training'") < text.index("'/api'")
