from __future__ import annotations

from types import SimpleNamespace

import pytest

from adsmod_ml.learning.training.manager import TrainingProcessRunner


def test_resume_validation_accepts_keras3_public_loss_state() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=object())
    TrainingProcessRunner.validate_resume_model(object(), model)


def test_resume_validation_rejects_missing_loss() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=None)
    with pytest.raises(ValueError, match="not compiled"):
        TrainingProcessRunner.validate_resume_model(object(), model)
