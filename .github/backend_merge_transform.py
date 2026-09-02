from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path.cwd()


def p(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    return p(rel).read_text(encoding="utf-8")


def write(rel: str, content: str) -> None:
    target = p(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace(rel: str, old: str, new: str, count: int = 1) -> None:
    text = read(rel)
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{rel}: expected {count} occurrence(s), found {actual}: {old[:100]!r}")
    write(rel, text.replace(old, new, count))


def regex_replace(rel: str, pattern: str, replacement: str, count: int = 1) -> None:
    text = read(rel)
    updated, actual = re.subn(pattern, replacement, text, count=count, flags=re.S)
    if actual != count:
        raise RuntimeError(f"{rel}: expected {count} regex replacement(s), found {actual}: {pattern[:100]!r}")
    write(rel, updated)


def delete(rel: str) -> None:
    target = p(rel)
    if target.exists():
        target.unlink()


write("app/backend/common/src/adsmod_common/capabilities.py", '''from __future__ import annotations

from pydantic import BaseModel


class FeatureCapabilities(BaseModel):
    datasets: bool
    nist: bool
    fitting: bool
    machine_learning: bool
    training: bool
    checkpoints: bool


class CapabilitiesResponse(BaseModel):
    version: str
    features: FeatureCapabilities


__all__ = ["CapabilitiesResponse", "FeatureCapabilities"]
''')

write("app/backend/common/src/adsmod_common/training_data.py", '''from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SnapshotReference:
    snapshot_id: str
    content_hash: str


@dataclass(frozen=True)
class SnapshotPayload:
    snapshot_id: str
    content_hash: str
    rows: tuple[dict[str, Any], ...]


class TrainingDataAccess(Protocol):
    def list_sources(self) -> list[dict[str, Any]]: ...

    def create_snapshot(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference: ...

    def create_snapshot_from_selections(
        self,
        selections: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference: ...

    def fetch_snapshot(self, snapshot_id: str) -> SnapshotPayload: ...


__all__ = ["SnapshotPayload", "SnapshotReference", "TrainingDataAccess"]
''')

cfg = "app/backend/common/src/adsmod_common/config.py"
replace(cfg, 'RuntimeMode = Literal["core", "core-ml"]\n\n\n', "")
regex_replace(cfg, r'class RuntimeConfig\(StrictModel\):.*?\n\n###############################################################################\nclass StorageConfig', '''class RuntimeConfig(StrictModel):
    host: str
    backend_port: int = Field(ge=1024, le=65535)
    frontend_port: int = Field(ge=1024, le=65535)

    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeConfig":
        if self.backend_port == self.frontend_port:
            raise ValueError("runtime.backend_port and runtime.frontend_port must be distinct")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("runtime.host must be a loopback address")
        return self


###############################################################################
class StorageConfig''')
regex_replace(cfg, r'\n###############################################################################\nclass SecurityConfig\(StrictModel\):.*?\n\n###############################################################################\nclass DatabaseConfig', '\n###############################################################################\nclass DatabaseConfig')
regex_replace(cfg, r'class AdsmodConfig\(StrictModel\):\n    version: Literal\["3\.0\.0"\]\n    runtime: RuntimeConfig\n    storage: StorageConfig\n    security: SecurityConfig\n    application: ApplicationConfig\n\n    # -------------------------------------------------------------------------\n    @model_validator\(mode="after"\)\n    def validate_mode_security\(self\) -> "AdsmodConfig":\n        if self\.runtime\.mode == "core-ml" and not self\.security\.internal_token_required:\n            raise ValueError\(\n                "security\.internal_token_required must be true in core-ml mode"\n            \)\n        return self', '''class AdsmodConfig(StrictModel):
    version: Literal["3.0.0"]
    runtime: RuntimeConfig
    storage: StorageConfig
    application: ApplicationConfig''')
replace(cfg, '    "RuntimeMode",\n', "")
replace(cfg, '    "SecurityConfig",\n', "")

replace("app/backend/pyproject.toml", "dependencies = []\n\n[dependency-groups]", 'dependencies = ["adsmod-core"]\n\n[project.optional-dependencies]\nml = ["adsmod-ml"]\n\n[dependency-groups]')
ml_pyproject = "app/backend/ml/pyproject.toml"
replace(ml_pyproject, '    "adsmod-common",\n', '    "adsmod-common",\n    "adsmod-core",\n')
text = read(ml_pyproject)
text = re.sub(r'^\s*"httpx[^"]*",\n', "", text, flags=re.M)
text = re.sub(r'^\s*"uvicorn[^"]*",\n', "", text, flags=re.M)
write(ml_pyproject, text)

config_file = p("app/resources/adsmod.json")
config = json.loads(config_file.read_text(encoding="utf-8"))
runtime = config["runtime"]
config["runtime"] = {"host": runtime["host"], "backend_port": runtime["core_port"], "frontend_port": runtime["frontend_port"]}
config.pop("security", None)
config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

write("app/backend/core/src/adsmod_core/services/training_data.py", '''from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from adsmod_common.config import AdsmodConfig
from adsmod_common.training_data import SnapshotPayload, SnapshotReference
from adsmod_core.api import SnapshotDatasetSelection
from adsmod_core.persistence.paths import resolve_storage_root
from adsmod_core.persistence.snapshots import SnapshotStore
from adsmod_core.repositories.database.initializer import prepare_database_for_startup
from adsmod_core.services.container import CoreServiceContainer


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class TrainingDataService:
    def __init__(self, container: CoreServiceContainer, *, owns_database: bool = False) -> None:
        self.container = container
        self.snapshot_store = SnapshotStore(container.database)
        self.owns_database = owns_database

    def close(self) -> None:
        if self.owns_database:
            self.container.database.dispose()

    def list_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        nist_count = int(self.container.nist_repository.count_nist_rows().get("single_component_rows", 0))
        if nist_count > 0:
            sources.append({"source": "nist", "dataset_name": "NIST ISODB", "display_name": "NIST Single Component", "row_count": nist_count, "dataset_id": None})
        sources.extend({"source": item["source"], "dataset_name": item["name"], "display_name": item["name"], "row_count": item["observation_count"], "dataset_id": item["id"]} for item in self.container.datasets.list_summaries() if item["source"] == "uploaded")
        return sources

    def create_snapshot(self, rows: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> SnapshotReference:
        record = self.snapshot_store.create(rows, metadata=metadata)
        return SnapshotReference(record.snapshot_id, record.content_hash)

    def create_snapshot_from_selections(self, selections: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> SnapshotReference:
        validated = [SnapshotDatasetSelection.model_validate(selection) for selection in selections]
        rows: list[dict[str, Any]] = []
        for selection in validated:
            if selection.source == "uploaded":
                rows.extend(self._uploaded_snapshot_rows(selection.dataset_name, selection.dataset_id))
            else:
                rows.extend(self._nist_snapshot_rows(selection.dataset_name))
        return self.create_snapshot(rows, metadata={**dict(metadata or {}), "selections": [selection.model_dump(mode="json") for selection in validated]})

    def fetch_snapshot(self, snapshot_id: str) -> SnapshotPayload:
        rows: list[dict[str, Any]] = []
        page_number = 1
        content_hash: str | None = None
        total_rows: int | None = None
        while total_rows is None or len(rows) < total_rows:
            page = self.snapshot_store.get_page(snapshot_id, page_number, 1000)
            if content_hash is None:
                content_hash = page.content_hash
                total_rows = page.total_rows
            elif page.content_hash != content_hash:
                raise RuntimeError("Snapshot content hash changed during read.")
            rows.extend(dict(row) for row in page.rows)
            page_number += 1
        payload = json.dumps(tuple(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        computed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if content_hash is None or computed_hash != content_hash:
            raise RuntimeError("Snapshot content hash verification failed.")
        return SnapshotPayload(snapshot_id, content_hash, tuple(rows))

    def _uploaded_snapshot_rows(self, dataset_name: str, dataset_id: int | None) -> list[dict[str, Any]]:
        summaries = self.container.datasets.list_summaries()
        dataset = next((item for item in summaries if item["source"] == "uploaded" and ((dataset_id is None and item["name"] == dataset_name) or item["id"] == dataset_id)), None)
        if dataset is None:
            raise LookupError(f"Uploaded dataset '{dataset_name}' does not exist.")
        frame = self.container.datasets.observation_frame(int(dataset["id"]))
        if frame.empty:
            raise ValueError(f"Uploaded dataset '{dataset_name}' contains no observations.")
        rows: list[dict[str, Any]] = []
        for index, raw_record in enumerate(frame.to_dict(orient="records")):
            record = {str(key): _json_safe(value) for key, value in raw_record.items()}
            experiment = str(record.get("experiment") or f"row-{index}")
            record.update({"filename": f"{dataset_name}:{experiment}", "temperature": record.get("temperature"), "adsorbent_name": record.get("adsorbent_name"), "adsorbate_name": record.get("adsorbate_name"), "pressure_units": "Pa", "adsorption_units": "mol/kg"})
            rows.append(record)
        return rows

    def _nist_snapshot_rows(self, dataset_name: str) -> list[dict[str, Any]]:
        adsorption, guests, _ = self.container.nist_repository.load_adsorption_datasets()
        if adsorption.empty:
            raise ValueError(f"NIST dataset '{dataset_name}' contains no observations.")
        guest_properties = {str(row.get("name", "")).strip().casefold(): row for row in guests.to_dict(orient="records")}
        rows: list[dict[str, Any]] = []
        for index, raw_record in enumerate(adsorption.to_dict(orient="records")):
            record = {str(key): _json_safe(value) for key, value in raw_record.items()}
            adsorbate = str(record.get("adsorbate") or "").strip()
            adsorbent = str(record.get("adsorbent") or "").strip()
            guest = guest_properties.get(adsorbate.casefold(), {})
            record.update({"filename": f"{dataset_name}:{record.get('external_key') or index}", "temperature": record.get("temperature_k"), "adsorbent_name": adsorbent, "adsorbate_name": adsorbate, "adsorbate_molecular_weight": _json_safe(guest.get("molecular_weight")), "adsorbate_SMILE": _json_safe(guest.get("smile_code")), "pressure_units": "Pa", "adsorption_units": "mol/kg"})
            rows.append(record)
        return rows


def open_training_data_service(config: AdsmodConfig) -> TrainingDataService:
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    prepare_database_for_startup(config.application.database, storage_root=storage_root)
    return TrainingDataService(CoreServiceContainer(config), owns_database=True)


__all__ = ["TrainingDataService", "open_training_data_service"]
''')

write("app/backend/core/src/adsmod_core/app.py", '''from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, get_args

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.units import UnitRegistry
from adsmod_common.version import __version__

from .common.constants import FASTAPI_DESCRIPTION, FASTAPI_TITLE
from .common.utils.logger import close_file_logging, configure_logging, logger
from .contracts.configuration import DisplayUnitCapabilities, FittingConfigurationResponse, NumericBounds, ParameterDefaults
from .contracts.fitting import FittingRequest
from .http.entrypoint import health_router, register_root_routes
from .http.routes import register_core_routes
from .persistence.paths import resolve_storage_root
from .repositories.database.initializer import prepare_database_for_startup
from .services.container import CoreServiceContainer
from .services.training_data import TrainingDataService


@dataclass
class ApplicationRuntime:
    config: AdsmodConfig
    container: CoreServiceContainer
    training_data: TrainingDataService
    machine_learning_available: bool = False
    machine_learning_reason: str | None = None
    ml_container: Any | None = None


def capabilities(request: Request) -> CapabilitiesResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    available = runtime.machine_learning_available
    return CapabilitiesResponse(version=__version__, features=FeatureCapabilities(datasets=True, nist=True, fitting=True, machine_learning=available, training=available, checkpoints=available))


def configuration(request: Request) -> FittingConfigurationResponse:
    config: AdsmodConfig = request.app.state.runtime.config
    fitting = config.application.fitting
    pressure_units = tuple(UnitRegistry.PRESSURE_TO_PA)
    uptake_units = tuple(dict.fromkeys(UnitRegistry.UPTAKE_ALIASES.values()))
    return FittingConfigurationResponse(supported_optimizers=tuple(get_args(FittingRequest.model_fields["optimizer"].annotation)), default_optimizer="trf", default_max_evaluations=fitting.default_max_iterations, max_evaluations_bounds=NumericBounds(minimum=10, maximum=fitting.max_iterations_upper_bound), weighting_options=tuple(get_args(FittingRequest.model_fields["weighting"].annotation)), default_weighting="unweighted", display_units=DisplayUnitCapabilities(pressure=pressure_units + ("1", "%"), uptake=uptake_units, default_pressure="bar", default_uptake="mmol/g"), parameter_defaults=ParameterDefaults(lower=fitting.default_parameter_min, upper=fitting.default_parameter_max, initial=fitting.default_parameter_initial))


def _build_system_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/system/capabilities", capabilities, methods=["GET"], response_model=CapabilitiesResponse, tags=["system"])
    router.add_api_route("/system/configuration", configuration, methods=["GET"], response_model=FittingConfigurationResponse, tags=["system"])
    return router


def _register_optional_ml(application: FastAPI, runtime: ApplicationRuntime) -> None:
    try:
        bootstrap = import_module("adsmod_ml.bootstrap")
        bootstrap.configure_environment()
        container_module = import_module("adsmod_ml.services.container")
        routes_module = import_module("adsmod_ml.http.routes")
        ml_container = container_module.MlServiceContainer(runtime.config, snapshot_access=runtime.training_data)
        routes_module.register_ml_routes(application, ml_container, prefix="/api/v1")
    except Exception as exc:  # noqa: BLE001
        runtime.machine_learning_available = False
        runtime.machine_learning_reason = str(exc)
        logger.info("Optional machine learning support is unavailable: %s", exc)
        return
    runtime.ml_container = ml_container
    runtime.machine_learning_available = True
    runtime.machine_learning_reason = None
    application.state.ml_container = ml_container


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    config: AdsmodConfig = application.state.config
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    configure_logging(storage_root / "logs")
    prepare_database_for_startup(config.application.database, storage_root=storage_root)
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        application.state.core_container.database.dispose()
        close_file_logging()


def create_app(config: AdsmodConfig) -> FastAPI:
    container = CoreServiceContainer(config)
    runtime = ApplicationRuntime(config=config, container=container, training_data=TrainingDataService(container))
    application = FastAPI(title=FASTAPI_TITLE, description=FASTAPI_DESCRIPTION, version=__version__, lifespan=app_lifespan)
    application.state.config = config
    application.state.ready = False
    application.state.core_container = container
    application.state.runtime = runtime
    application.add_middleware(CORSMiddleware, allow_origins=[f"http://{config.runtime.host}:{config.runtime.frontend_port}", f"http://127.0.0.1:{config.runtime.frontend_port}", f"http://localhost:{config.runtime.frontend_port}"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    application.include_router(health_router)
    application.include_router(_build_system_router(), prefix="/api/v1")
    register_core_routes(application, container, prefix="/api/v1")
    _register_optional_ml(application, runtime)
    register_root_routes(application)
    return application


def create_app_from_path(config_path: str | Path) -> FastAPI:
    return create_app(load_config(config_path))


__all__ = ["app_lifespan", "create_app", "create_app_from_path"]
''')

replace("app/backend/core/src/adsmod_core/cli.py", "port=runtime.core_port", "port=runtime.backend_port")
replace("app/backend/core/src/adsmod_core/http/entrypoint.py", 'service="core"', 'service="backend"', count=2)
replace("app/backend/core/src/adsmod_core/common/constants.py", 'FASTAPI_TITLE = "ADSMOD Model Fitting Backend"', 'FASTAPI_TITLE = "ADSMOD Backend"')

write("app/backend/ml/src/adsmod_ml/services/container.py", '''from __future__ import annotations

from adsmod_common.config import AdsmodConfig
from adsmod_common.paths import resolve_checkpoint_root, resolve_storage_root
from adsmod_common.training_data import TrainingDataAccess
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.learning.training.manager import TrainingManager
from adsmod_ml.services.jobs import JobManager
from adsmod_ml.services.training import TrainingJobRunner, TrainingService, TrainingSession


class MlServiceContainer:
    def __init__(self, config: AdsmodConfig, *, snapshot_access: TrainingDataAccess) -> None:
        self.config = config
        self.snapshot_access = snapshot_access
        self.artifact_root = resolve_storage_root(config) / "training"
        self.checkpoints_dir = resolve_checkpoint_root(config)
        self.job_manager = JobManager(logger=logger)
        self.training_manager = TrainingManager(config, snapshot_access=self.snapshot_access, artifact_root=self.artifact_root, checkpoints_dir=self.checkpoints_dir)
        self.training_session = TrainingSession(training_manager=self.training_manager)
        self.training_job_runner = TrainingJobRunner(session=self.training_session, job_manager=self.job_manager, training_manager=self.training_manager, config=config, process_context={"config_payload": config.model_dump(mode="json"), "artifact_root": str(self.artifact_root), "checkpoints_dir": str(self.checkpoints_dir)})
        self.training_service = TrainingService(config=config, snapshot_access=self.snapshot_access, artifact_root=self.artifact_root, checkpoints_dir=self.checkpoints_dir, job_manager=self.job_manager, training_manager=self.training_manager, training_session=self.training_session, training_job_runner=self.training_job_runner)


__all__ = ["MlServiceContainer"]
''')

manager = "app/backend/ml/src/adsmod_ml/learning/training/manager.py"
replace(manager, "from adsmod_ml.clients.core_client import CoreSnapshotClient\n", "from adsmod_common.training_data import TrainingDataAccess\n")
replace(manager, "snapshot_client: CoreSnapshotClient,", "snapshot_access: TrainingDataAccess,")
replace(manager, "TrainingDataSerializer(snapshot_client, artifact_root)", "TrainingDataSerializer(snapshot_access, artifact_root)")
regex_replace(manager, r'def run_training_process\(\n    configuration: dict\[str, Any\] \| None,\n    checkpoint: str \| None = None,\n    additional_epochs: int = 0,\n    worker: Any \| None = None,\n    core_base_url: str = "",\n    internal_token: str = "",\n    artifact_root: str = "",\n    checkpoints_dir: str = "",\n\) -> None:\n.*?\n\n###############################################################################\nclass TrainingManager:', '''def run_training_process(
    configuration: dict[str, Any] | None,
    checkpoint: str | None = None,
    additional_epochs: int = 0,
    worker: Any | None = None,
    config_payload: dict[str, Any] | None = None,
    artifact_root: str = "",
    checkpoints_dir: str = "",
) -> None:
    result_queue = getattr(worker, "result_queue", None)
    stop_event = getattr(worker, "stop_event", None)
    snapshot_access = None
    try:
        if stop_event is not None and stop_event.is_set():
            put_worker_result(result_queue, {"result": {}})
            return
        if config_payload is None or not artifact_root or not checkpoints_dir:
            raise ValueError("ML process runtime configuration and paths are required.")
        from adsmod_core.services.training_data import open_training_data_service
        runtime_config = AdsmodConfig.model_validate(config_payload)
        snapshot_access = open_training_data_service(runtime_config)
        runner = TrainingProcessRunner(worker=worker, snapshot_access=snapshot_access, artifact_root=Path(artifact_root), checkpoints_dir=Path(checkpoints_dir))
        if checkpoint:
            runner.log(f"Resuming training from checkpoint {checkpoint} for {additional_epochs} additional epochs.")
            runner.resume_training(checkpoint, additional_epochs)
            put_worker_result(result_queue, {"result": {"success": True, "checkpoint": checkpoint}})
            return
        if configuration is None:
            raise ValueError("Training configuration is required.")
        runner.log("Starting training session.")
        runner.start_training(configuration)
        put_worker_result(result_queue, {"result": {"success": True}})
    except WorkerInterrupted:
        put_worker_result(result_queue, {"result": {}})
    except Exception as exc:  # noqa: BLE001
        put_worker_result(result_queue, {"error": str(exc)})
    finally:
        if snapshot_access is not None:
            snapshot_access.close()


###############################################################################
class TrainingManager:''')
regex_replace(manager, r'    def __init__\(\n        self,\n        config: AdsmodConfig,\n        \*,\n        snapshot_client: CoreSnapshotClient \| None = None,\n        artifact_root: Path \| None = None,\n        checkpoints_dir: Path \| None = None,\n    \) -> None:\n        client = snapshot_client or CoreSnapshotClient\.from_config\(config\)\n        resolved_artifact_root = \(\n            artifact_root or resolve_storage_root\(config\) / "training"\n        \)\n        resolved_checkpoints_dir = checkpoints_dir or resolve_checkpoint_root\(config\)\n        self\.state = TrainingState\(\)\n        self\.data_serializer = TrainingDataSerializer\(client, resolved_artifact_root\)\n        self\.model_serializer = ModelSerializer\(resolved_checkpoints_dir\)', '''    def __init__(
        self,
        config: AdsmodConfig,
        *,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        resolved_artifact_root = artifact_root or resolve_storage_root(config) / "training"
        resolved_checkpoints_dir = checkpoints_dir or resolve_checkpoint_root(config)
        self.state = TrainingState()
        self.data_serializer = TrainingDataSerializer(snapshot_access, resolved_artifact_root)
        self.model_serializer = ModelSerializer(resolved_checkpoints_dir)''')

serialization = "app/backend/ml/src/adsmod_ml/learning/serialization/training.py"
replace(serialization, "from adsmod_ml.clients.core_client import CoreSnapshotClient\n", "from adsmod_common.training_data import TrainingDataAccess\n")
replace(serialization, "self, snapshot_client: CoreSnapshotClient, artifact_root: Path", "self, snapshot_access: TrainingDataAccess, artifact_root: Path")
text = read(serialization).replace("snapshot_client", "snapshot_access").replace("Core-owned immutable training snapshots", "backend-owned immutable training snapshots").replace("Core snapshot hash", "Snapshot hash")
write(serialization, text)

composition = "app/backend/ml/src/adsmod_ml/services/data/composition.py"
replace(composition, "from adsmod_ml.clients.core_client import CoreSnapshotClient\n", "from adsmod_common.training_data import TrainingDataAccess\n")
text = read(composition).replace("Core's immutable snapshot API", "the backend's immutable snapshot service").replace("snapshot_client: CoreSnapshotClient", "snapshot_access: TrainingDataAccess").replace("snapshot_client", "snapshot_access").replace("Core snapshot is missing", "Snapshot is missing")
write(composition, text)

training = "app/backend/ml/src/adsmod_ml/services/training.py"
replace(training, "from adsmod_ml.clients.core_client import CoreSnapshotClient\n", "from adsmod_common.training_data import TrainingDataAccess\n")
text = read(training).replace("snapshot_client: CoreSnapshotClient", "snapshot_access: TrainingDataAccess").replace("snapshot_client", "snapshot_access")
text = re.sub(r'\n        internal_token: str,', "", text)
text = re.sub(r'\n        self\.internal_token = internal_token', "", text)
write(training, text)

for rel in ("app/backend/ml/src/adsmod_ml/app.py", "app/backend/ml/src/adsmod_ml/cli.py", "app/backend/ml/src/adsmod_ml/clients/core_client.py", "app/backend/ml/src/adsmod_ml/http/entrypoint.py"):
    delete(rel)

write("app/client/proxy.conf.cjs", '''const fs = require('node:fs');
const path = require('node:path');

const repositoryRoot = path.resolve(__dirname, '../..');
const canonicalConfig = JSON.parse(fs.readFileSync(path.join(repositoryRoot, 'app', 'resources', 'adsmod.json'), 'utf8'));
const runtime = canonicalConfig.runtime;
const backendTarget = `http://${runtime.host}:${Number(runtime.backend_port)}`;

module.exports = {
  '/api/v1': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' },
  '/health': { target: backendTarget, changeOrigin: true, secure: false, logLevel: 'warn' }
};
''')

write("app/client/src/app/services/system.service.ts", '''import { API_BASE_URL } from '../core/config/api-base-url';
import type { FittingConfiguration } from '../models/fitting.model';
import type { TrainingConfiguration } from '../models/training.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

export interface ApplicationCapabilities {
    version: string;
    features: { datasets: boolean; nist: boolean; fitting: boolean; machine_learning: boolean; training: boolean; checkpoints: boolean; };
}
export interface ServiceHealth { service: string; version: string; state: string; }
export interface ServiceResult<T> { data: T | null; error: string | null; }
const asRecord = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' ? value as Record<string, unknown> : null;
const readJson = async <T>(url: string): Promise<ServiceResult<T>> => {
    try {
        const response = await fetchWithTimeout(url, { method: 'GET' }, HTTP_TIMEOUT);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) return { data: null, error: extractErrorMessage(response, body) };
        if (!asRecord(body)) return { data: null, error: `Invalid response from ${url}.` };
        return { data: body as T, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
};
let capabilitiesRequest: Promise<ServiceResult<ApplicationCapabilities>> | null = null;
export const fetchApplicationCapabilities = (refresh = false): Promise<ServiceResult<ApplicationCapabilities>> => {
    if (refresh || capabilitiesRequest === null) capabilitiesRequest = readJson<ApplicationCapabilities>(`${API_BASE_URL}/system/capabilities`);
    return capabilitiesRequest;
};
export const machineLearningAvailable = async (): Promise<boolean> => (await fetchApplicationCapabilities()).data?.features.machine_learning === true;
export const fetchFittingConfiguration = (): Promise<ServiceResult<FittingConfiguration>> => readJson<FittingConfiguration>(`${API_BASE_URL}/system/configuration`);
export const fetchTrainingConfiguration = (): Promise<ServiceResult<TrainingConfiguration>> => readJson<TrainingConfiguration>(`${API_BASE_URL}/training/configuration`);
export const fetchBackendReadiness = (): Promise<ServiceResult<ServiceHealth>> => readJson<ServiceHealth>('/health/ready');
''')

write("app/client/src/app/core/guards/machine-learning.guard.ts", '''import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { machineLearningAvailable } from '../../services/system.service';

export const machineLearningGuard: CanActivateFn = async () => {
    const router = inject(Router);
    return await machineLearningAvailable() ? true : router.createUrlTree(['/datasets']);
};
''')

routes = "app/client/src/app/app.routes.ts"
replace(routes, "import { CoreShellComponent } from './layout/core-shell.component';\n", "import { machineLearningGuard } from './core/guards/machine-learning.guard';\nimport { CoreShellComponent } from './layout/core-shell.component';\n")
replace(routes, "{ path: 'training', pathMatch: 'full', redirectTo: 'training/processing' },", "{ path: 'training', pathMatch: 'full', canActivate: [machineLearningGuard], redirectTo: 'training/processing' },")
replace(routes, "                path: 'training/:view',\n                loadComponent:", "                path: 'training/:view',\n                canActivate: [machineLearningGuard],\n                loadComponent:")

shell = "app/client/src/app/layout/core-shell.component.ts"
replace(shell, "import { fetchCoreCapabilities, fetchCoreReadiness, fetchMlReadiness } from '../services/system.service';", "import { fetchApplicationCapabilities, fetchBackendReadiness } from '../services/system.service';")
replace(shell, "Training requires the optional ML service.", "Training requires the optional machine learning dependencies.")
replace(shell, '''                    <a class="console-nav-item" routerLink="/training" routerLinkActive="active">
                        <span class="console-nav-icon" aria-hidden="true">✺</span>
                        <span>Training</span>
                    </a>''', '''                    @if (machineLearningAvailable()) {
                        <a class="console-nav-item" routerLink="/training" routerLinkActive="active">
                            <span class="console-nav-icon" aria-hidden="true">✺</span>
                            <span>Training</span>
                        </a>
                    }''')
regex_replace(shell, r'        <div class="console-status-bar" aria-label="Service status" aria-live="polite">.*?        </div>\n        @if \(helpOpen\(\)\)', '''        <div class="console-status-bar" aria-label="Backend status" aria-live="polite">
            <div class="console-status-item"><span class="service-dot core" [class.offline]="backendStatus() === 'Offline'" aria-hidden="true"></span><strong>Backend</strong><em>{{ backendStatus() }}</em></div>
        </div>
        @if (helpOpen())''')
replace(shell, "    protected readonly coreServiceStatus = signal<'Checking' | 'Online' | 'Offline'>('Checking');\n    protected readonly mlServiceStatus = signal<'Checking' | 'Online' | 'Unavailable'>('Checking');", "    protected readonly backendStatus = signal<'Checking' | 'Online' | 'Offline'>('Checking');\n    protected readonly machineLearningAvailable = signal(false);")
write(shell, read(shell).replace("refreshServiceStatus", "refreshBackendStatus"))
regex_replace(shell, r'    private async refreshBackendStatus\(\): Promise<void> \{\n.*?\n    \}\n\n    @HostListener', '''    private async refreshBackendStatus(): Promise<void> {
        const [capabilities, readiness] = await Promise.all([fetchApplicationCapabilities(true), fetchBackendReadiness()]);
        const backendOnline = capabilities.data !== null && readiness.data?.state === 'ready';
        this.backendStatus.set(backendOnline ? 'Online' : 'Offline');
        this.machineLearningAvailable.set(capabilities.data?.features.machine_learning === true);
    }

    @HostListener''')

ps1 = "start_on_windows.ps1"
regex_replace(ps1, r'function Import-Settings \{.*?\n\}\n\nfunction Set-RuntimeEnvironment', '''function Import-Settings {
    if (-not (Test-Path -LiteralPath $ConfigFile)) { throw "Missing canonical configuration: $ConfigFile" }
    $canonical = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    if (-not $canonical.runtime -or -not $canonical.storage) { throw "Canonical configuration is missing runtime or storage settings: $ConfigFile" }
    $host = [string]$canonical.runtime.host
    if ([string]::IsNullOrWhiteSpace($host)) { throw "runtime.host must be configured." }
    $script:ResourcesDir = Resolve-CanonicalPath ([string]$canonical.storage.root)
    $script:LogDir = Join-Path $script:ResourcesDir "logs"
    $script:CheckpointsDir = Join-Path $script:ResourcesDir "checkpoints"
    return [pscustomobject]@{ Host = $host; BackendPort = [int]$canonical.runtime.backend_port; FrontendPort = [int]$canonical.runtime.frontend_port }
}

function Set-RuntimeEnvironment''')
regex_replace(ps1, r'function Sync-Dependencies \{.*?\n\}\n\nfunction Test-DependenciesReady', '''function Sync-Dependencies {
    param([switch]$BuildFrontend, [switch]$RuntimesReady, [ValidateSet('Standard', 'Development')][string]$InstallationType = 'Standard', [ValidateSet('Base', 'ML')][string]$FeatureSet = 'Base')
    Import-Settings | Out-Null
    if (-not $RuntimesReady) { Initialize-Runtimes }
    Set-RuntimeEnvironment
    Write-Step "Syncing Python dependencies ($FeatureSet)"
    Push-Location $BackendDir
    try {
        $arguments = @('sync', '--locked', '--python', $PythonExe)
        if ($FeatureSet -eq 'ML') { $arguments += '--extra', 'ml' }
        if ($InstallationType -eq 'Development') { $arguments += '--group', 'dev' } else { $arguments += '--no-dev' }
        & $UvExe @arguments
        Assert-LastExitCode "uv sync"
    } finally { Pop-Location }
    if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Backend virtual-environment Python was not created at $VenvPython." }
    Write-Ok "Python dependencies are ready."
    Sync-FrontendDependencies -BuildFrontend:$BuildFrontend
}

function Test-DependenciesReady''')
regex_replace(ps1, r'function Start-Application \{.*?\n\}\n\nfunction Install-UpdateDependencies', '''function Start-Application {
    $settings = Import-Settings
    Set-RuntimeEnvironment
    if (-not (Test-DependenciesReady)) { Write-Step "Required application environments or frontend build output are missing or unusable; repairing the base installation."; Sync-Dependencies -BuildFrontend -FeatureSet Base } else { Write-Ok "Application environments are ready; skipped dependency installation." }
    Set-RuntimeEnvironment
    $backendPort = $settings.BackendPort
    $uiPort = $settings.FrontendPort
    Stop-ListenerOnPort -Port $backendPort
    Stop-ListenerOnPort -Port $uiPort
    $backendArguments = @('-m', 'adsmod_core.cli', '--config', $ConfigFile)
    Write-Step "Starting ADSMOD backend"
    $backendProcess = Start-Process -FilePath $VenvPython -ArgumentList $backendArguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    $healthUrl = "http://$($settings.Host):$($settings.BackendPort)/health/ready"
    Write-Step "Waiting for backend readiness at $healthUrl"
    try { Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60 } catch { if ($backendProcess -and -not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id -Force }; throw }
    $backendPid = Get-ListenerPid -Port $backendPort
    Write-Step "Starting frontend preview"
    $frontendProcess = Start-Process -FilePath $NpmCmd -ArgumentList @('run', 'preview', '--', '--host', $settings.Host, '--port', $settings.FrontendPort) -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
    $frontendUrl = "http://$($settings.Host):$($settings.FrontendPort)"
    try { Wait-ForHealth -Url $frontendUrl -TimeoutSeconds 60 } catch { if (-not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id -Force }; if ($backendProcess -and -not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id -Force }; throw }
    $frontendPid = Get-ListenerPid -Port $uiPort
    Start-Process $frontendUrl
    Write-Host ""
    Write-Ok "ADSMOD started successfully."
    Write-Host "Backend: $healthUrl (PID $backendPid)" -ForegroundColor Green
    Write-Host "Frontend: $frontendUrl (PID $frontendPid)" -ForegroundColor Green
}

function Install-UpdateDependencies''')
regex_replace(ps1, r'function Install-UpdateDependencies \{.*?\n\}\n\nfunction Rebuild-Frontend', '''function Install-UpdateDependencies {
    Initialize-Runtimes
    Write-Ok "Portable runtimes ready."
    $installationType = Read-InstallationType
    $featureSet = Read-FeatureSet
    Sync-Dependencies -BuildFrontend -RuntimesReady -InstallationType $installationType -FeatureSet $featureSet
    Remove-UvCache
    Write-Ok "Dependencies installed and frontend built successfully."
}

function Rebuild-Frontend''')
read_install = '''function Read-InstallationType {
    Write-Host "  [1] Development - include Ruff, Pyright, and pytest"
    Write-Host "  [2] Standard    - install runtime dependencies only"
    $selection = (Read-Host "  Select installation profile [1-2]").Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw "Invalid installation profile. Enter 1 for Development or 2 for Standard." }
    }
}
'''
replace(ps1, read_install, read_install + '''
function Read-FeatureSet {
    Write-Host "  [1] Base - core ADSMOD functionality only"
    Write-Host "  [2] ML   - base application plus optional machine learning dependencies"
    $selection = (Read-Host "  Select feature set [1-2]").Trim()
    switch ($selection) { '1' { return 'Base' }; '2' { return 'ML' }; default { throw "Invalid feature set. Enter 1 for Base or 2 for ML." } }
}
''')

delete("app/backend/openapi/core.json")
delete("app/backend/openapi/ml.json")
subprocess.run(["uv", "lock", "--project", "app/backend"], check=True)
subprocess.run(["uv", "run", "--project", "app/backend", "--group", "dev", "python", "app/scripts/generate_config_schema.py"], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
