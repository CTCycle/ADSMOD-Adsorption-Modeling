from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

import uuid
from dataclasses import dataclass

import pytest

from ADSMOD.server.schemas.training import TrainingConfigRequest
from ADSMOD.server.common.constants import CHECKPOINTS_PATH
from ADSMOD.tests.backend.performance.dataset import (
    SyntheticDatasetSpec,
    clear_synthetic_training_dataset,
    create_and_save_synthetic_training_dataset,
)
from ADSMOD.tests.backend.performance.harness import (
    list_checkpoint_folders,
    remove_checkpoint_folders,
    run_training_with_metrics,
)


###############################################################################
@dataclass
class TrainingScenario:
    name: str
    sample_count: int
    series_length: int
    smile_length: int
    batch_size: int
    epochs: int
    num_encoders: int
    num_attention_heads: int
    embedding_size: int
    adsorbent_count: int
    smile_vocab_size: int
    validation_fraction: float
    seed: int


###############################################################################
def build_training_configuration(
    scenario: TrainingScenario,
    dataset_label: str,
    dataset_hash: str | None,
) -> dict[str, object]:
    request = TrainingConfigRequest(
        sample_size=1.0,
        validation_size=scenario.validation_fraction,
        batch_size=scenario.batch_size,
        shuffle_dataset=True,
        dataset_label=dataset_label,
        dataset_hash=dataset_hash,
        selected_model="SCADS Series",
        dropout_rate=0.1,
        num_attention_heads=scenario.num_attention_heads,
        num_encoders=scenario.num_encoders,
        molecular_embedding_size=scenario.embedding_size,
        epochs=scenario.epochs,
        use_device_GPU=False,
        use_mixed_precision=False,
        use_lr_scheduler=True,
        initial_lr=1e-4,
        target_lr=1e-5,
        constant_steps=2,
        decay_steps=4,
        save_checkpoints=True,
        checkpoints_frequency=10,
        custom_name=f"perf_{scenario.name}",
    )
    configuration = request.model_dump()
    configuration["dataset_label"] = dataset_label
    configuration["dataset_hash"] = dataset_hash
    configuration["training_seed"] = scenario.seed
    return configuration


# -------------------------------------------------------------------------
def resolve_perf_limits(strict_mode: bool) -> dict[str, float]:
    if strict_mode:
        return {
            "timeout_seconds": float(
                os.getenv("PERF_TEST_STRICT_TIMEOUT_SECONDS", "40")
            ),
            "max_rss_mb": float(os.getenv("PERF_TEST_STRICT_MAX_RSS_MB", "1500")),
            "max_growth_mb": float(os.getenv("PERF_TEST_STRICT_MAX_GROWTH_MB", "800")),
            "baseline_epoch": float(os.getenv("PERF_TEST_STRICT_BASELINE_EPOCH", "1")),
            "poll_interval_seconds": float(
                os.getenv("PERF_TEST_STRICT_POLL_INTERVAL_SECONDS", "0.1")
            ),
            "stop_grace_seconds": float(
                os.getenv("PERF_TEST_STRICT_STOP_GRACE_SECONDS", "6")
            ),
        }
    return {
        "timeout_seconds": float(os.getenv("PERF_TEST_TIMEOUT_SECONDS", "70")),
        "max_rss_mb": float(os.getenv("PERF_TEST_MAX_RSS_MB", "2200")),
        "max_growth_mb": float(os.getenv("PERF_TEST_MAX_GROWTH_MB", "1500")),
        "baseline_epoch": float(os.getenv("PERF_TEST_BASELINE_EPOCH", "1")),
        "poll_interval_seconds": float(
            os.getenv("PERF_TEST_POLL_INTERVAL_SECONDS", "0.1")
        ),
        "stop_grace_seconds": float(os.getenv("PERF_TEST_STOP_GRACE_SECONDS", "8")),
    }


# -------------------------------------------------------------------------
def scenario_matrix() -> list[TrainingScenario]:
    return [
        TrainingScenario(
            name="small_bs8",
            sample_count=36,
            series_length=12,
            smile_length=16,
            batch_size=8,
            epochs=2,
            num_encoders=1,
            num_attention_heads=1,
            embedding_size=64,
            adsorbent_count=4,
            smile_vocab_size=12,
            validation_fraction=0.2,
            seed=11,
        ),
        TrainingScenario(
            name="medium_bs16",
            sample_count=72,
            series_length=14,
            smile_length=18,
            batch_size=16,
            epochs=2,
            num_encoders=1,
            num_attention_heads=2,
            embedding_size=64,
            adsorbent_count=6,
            smile_vocab_size=14,
            validation_fraction=0.2,
            seed=17,
        ),
        TrainingScenario(
            name="medium_bs32_enc2",
            sample_count=72,
            series_length=14,
            smile_length=18,
            batch_size=32,
            epochs=2,
            num_encoders=2,
            num_attention_heads=2,
            embedding_size=96,
            adsorbent_count=6,
            smile_vocab_size=14,
            validation_fraction=0.2,
            seed=23,
        ),
    ]


@pytest.mark.parametrize("scenario", scenario_matrix(), ids=lambda s: s.name)
def test_training_pipeline_performance(scenario: TrainingScenario) -> None:
    strict_mode = os.getenv("PERF_TEST_STRICT") == "1"
    limits = resolve_perf_limits(strict_mode)
    dataset_label = f"perf_{scenario.name}_{uuid.uuid4().hex[:8]}"

    metadata = None
    created_checkpoints: set[str] = set()

    try:
        spec = SyntheticDatasetSpec(
            sample_count=scenario.sample_count,
            series_length=scenario.series_length,
            smile_length=scenario.smile_length,
            adsorbent_count=scenario.adsorbent_count,
            smile_vocab_size=scenario.smile_vocab_size,
            validation_fraction=scenario.validation_fraction,
            seed=scenario.seed,
            dataset_label=dataset_label,
        )
        metadata = create_and_save_synthetic_training_dataset(spec)
        configuration = build_training_configuration(
            scenario, dataset_label, metadata.dataset_hash
        )

        checkpoints_before = list_checkpoint_folders(CHECKPOINTS_PATH)

        result = run_training_with_metrics(
            configuration=configuration,
            timeout_seconds=limits["timeout_seconds"],
            poll_interval_seconds=limits["poll_interval_seconds"],
            stop_grace_seconds=limits["stop_grace_seconds"],
            baseline_epoch=int(limits["baseline_epoch"]),
        )

        checkpoints_after = list_checkpoint_folders(CHECKPOINTS_PATH)
        created_checkpoints = checkpoints_after - checkpoints_before

        assert not result.timed_out, (
            "Training timed out. "
            f"scenario={scenario.name} runtime={result.runtime_seconds:.2f}s "
            f"timeout={limits['timeout_seconds']}s"
        )
        assert result.error is None, (
            "Training failed. "
            f"scenario={scenario.name} error={result.error} exit_code={result.exit_code}"
        )

        peak_mb = result.peak_rss_bytes / (1024 * 1024)
        base_mb = result.base_rss_bytes / (1024 * 1024)
        growth_mb = max(0.0, peak_mb - base_mb)

        assert result.runtime_seconds <= limits["timeout_seconds"], (
            "Training exceeded runtime ceiling. "
            f"scenario={scenario.name} runtime={result.runtime_seconds:.2f}s "
            f"limit={limits['timeout_seconds']}s"
        )
        assert peak_mb <= limits["max_rss_mb"], (
            "Training peak RSS exceeded ceiling. "
            f"scenario={scenario.name} peak_rss={peak_mb:.1f}MB "
            f"limit={limits['max_rss_mb']}MB"
        )
        assert growth_mb <= limits["max_growth_mb"], (
            "Training memory growth exceeded ceiling. "
            f"scenario={scenario.name} base_rss={base_mb:.1f}MB "
            f"peak_rss={peak_mb:.1f}MB growth={growth_mb:.1f}MB "
            f"limit={limits['max_growth_mb']}MB"
        )
    finally:
        if created_checkpoints:
            remove_checkpoint_folders(CHECKPOINTS_PATH, created_checkpoints)
        clear_synthetic_training_dataset(dataset_label)
