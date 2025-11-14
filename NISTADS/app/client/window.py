from __future__ import annotations

from collections.abc import Callable
from typing import cast

from matplotlib.figure import Figure
import pandas as pd

from NISTADS.app.utils.variables import env_variables
from functools import partial

from PySide6.QtCore import QFile, QIODevice, Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QAction, QPainter, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
)
from qt_material import apply_stylesheet

from NISTADS.app.client.dialogs import LoadConfigDialog, SaveConfigDialog
from NISTADS.app.client.events import (
    Any,
    DatasetEvents,
    GraphicsHandler,
    ModelEvents,
    ValidationEvents,
)
from NISTADS.app.client.workers import ProcessWorker, ThreadWorker
from NISTADS.app.utils.configuration import Configuration
from NISTADS.app.utils.logger import logger
from NISTADS.app.utils.repository.database import database


###############################################################################
def apply_style(app: QApplication) -> QApplication:
    theme = "dark_yellow"
    extra = {"density_scale": "-1"}
    apply_stylesheet(app, theme=f"{theme}.xml", extra=extra)

    # Make % text visible/centered for ALL progress bars
    app.setStyleSheet(
        app.styleSheet()
        + """
    QProgressBar {
        text-align: center;  /* align percentage to the center */
        color: black;        /* black text for yellow bar */
        font-weight: bold;   /* bold percentage */        
    }
    """
    )

    return app


###############################################################################
class MainWindow:
    def __init__(self, ui_file_path: str) -> None:
        super().__init__()
        loader = QUiLoader()
        ui_file = QFile(ui_file_path)
        ui_file.open(QIODevice.OpenModeFlag.ReadOnly)
        self.main_win = cast(QMainWindow, loader.load(ui_file))
        ui_file.close()

        # Checkpoint & metrics state
        self.checkpoints_list: QComboBox
        self.selected_checkpoint = None
        self.selected_metrics = {"dataset": [], "model": []}

        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()

        # set thread pool for the workers
        self.threadpool = QThreadPool.globalInstance()
        self.worker: ThreadWorker | ProcessWorker | None = None

        # initialize database
        database.initialize_database()

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.dataset_handler = DatasetEvents(self.configuration)
        self.validation_handler = ValidationEvents(self.configuration)
        self.model_handler = ModelEvents(self.configuration)

        # setup UI elements
        self.set_states()
        self.widgets = {}
        self.setup_configuration(
            [
                # actions
                (QAction, "actionLoadConfig", "load_configuration_action"),
                (QAction, "actionSaveConfig", "save_configuration_action"),
                (QAction, "actionDeleteData", "delete_data_action"),
                (QAction, "actionExportData", "export_data_action"),
                # out of tab widgets
                (QProgressBar, "progressBar", "progress_bar"),
                (QPushButton, "stopThread", "stop_thread"),
                (QCheckBox, "deviceGPU", "use_device_GPU"),
                # 1. dataset tab page
                # dataset fetching group
                (QDoubleSpinBox, "guestFraction", "guest_fraction"),
                (QDoubleSpinBox, "hostFraction", "host_fraction"),
                (QDoubleSpinBox, "expFraction", "experiments_fraction"),
                (QSpinBox, "parallelTasks", "parallel_tasks"),
                (QPushButton, "collectAdsData", "collect_adsorption_data"),
                (QPushButton, "retrieveGuestProperties", "retrieve_guest_properties"),
                (QPushButton, "retrieveHostProperties", "retrieve_host_properties"),
                # dataset evaluation and processing group
                (QSpinBox, "seed", "seed"),
                (QSpinBox, "splitSeed", "split_seed"),
                (QDoubleSpinBox, "sampleSize", "sample_size"),
                (QDoubleSpinBox, "validationSize", "validation_size"),
                (QSpinBox, "maxPoints", "max_measurements"),
                (QSpinBox, "minPoints", "min_measurements"),
                (QSpinBox, "smileSeqSize", "SMILE_sequence_size"),
                (QSpinBox, "maxPressure", "max_pressure"),
                (QDoubleSpinBox, "maxUptake", "max_uptake"),
                (QPushButton, "buildMLDataset", "build_ML_dataset"),
                (QCheckBox, "adsIsothermCluster", "experiments_clustering"),
                (QPushButton, "evaluateDataset", "evaluate_dataset"),
                # 2. model tab page
                # dataset settings group
                (QCheckBox, "setShuffle", "use_shuffle"),
                (QSpinBox, "shuffleSize", "shuffle_size"),
                # device settings group
                (QSpinBox, "deviceID", "device_ID"),
                (QSpinBox, "numWorkers", "num_workers"),
                # training settings group
                (QSpinBox, "numEpochs", "epochs"),
                (QSpinBox, "batchSize", "batch_size"),
                (QSpinBox, "trainSeed", "train_seed"),
                (QCheckBox, "mixedPrecision", "use_mixed_precision"),
                (QCheckBox, "compileJIT", "use_JIT_compiler"),
                (QComboBox, "backendJIT", "jit_backend"),
                (QCheckBox, "runTensorboard", "use_tensorboard"),
                (QCheckBox, "realTimeHistory", "real_time_history_callback"),
                (QCheckBox, "saveCheckpoints", "save_checkpoints"),
                (QSpinBox, "saveCPFrequency", "checkpoints_frequency"),
                # RL scheduler settings group
                (QCheckBox, "useScheduler", "LR_scheduler"),
                (QDoubleSpinBox, "initialLearningRate", "initial_LR"),
                (QDoubleSpinBox, "targetLearningRate", "target_LR"),
                (QSpinBox, "constantSteps", "constant_steps"),
                (QSpinBox, "decaySteps", "decay_steps"),
                # model settings group
                (QComboBox, "modelType", "selected_model"),
                (QDoubleSpinBox, "dropoutRate", "dropout_rate"),
                (QSpinBox, "attentionHeads", "num_attention_heads"),
                (QSpinBox, "numEncoders", "num_encoders"),
                (QSpinBox, "molEmbeddingDims", "molecular_embedding_size"),
                # session settings group
                (QSpinBox, "numAdditionalEpochs", "additional_epochs"),
                (QPushButton, "startTraining", "start_training"),
                (QPushButton, "resumeTraining", "resume_training"),
                # model inference and evaluation
                (QPushButton, "refreshCheckpoints", "refresh_checkpoints"),
                (QComboBox, "checkpointsList", "checkpoints_list"),
                (QSpinBox, "evalSamples", "num_evaluation_samples"),
                (QPushButton, "evaluateModel", "model_evaluation"),
                (QPushButton, "checkpointSummary", "checkpoints_summary"),
                (QCheckBox, "evalReport", "get_evaluation_report"),
                (QCheckBox, "adsIsothermsComparison", "get_prediction_quality"),
                (QSpinBox, "inferenceBatchSize", "inference_batch_size"),
                (QPushButton, "loadDataSources", "load_data_sources"),
                (QPushButton, "predictAdsorption", "predict_adsorption"),
                # 3. Viewer tab
                (QRadioButton, "viewTrainMetrics", "train_metrics_view"),
                (QPushButton, "previousImg", "previous_image"),
                (QPushButton, "nextImg", "next_image"),
                (QPushButton, "clearImg", "clear_images"),
            ]
        )

        self.connect_signals(
            [
                # actions
                ("save_configuration_action", "triggered", self.save_configuration),
                ("load_configuration_action", "triggered", self.load_configuration),
                ("delete_data_action", "triggered", self.delete_all_data),
                ("export_data_action", "triggered", self.export_all_data),
                # out of tab widgets
                ("stop_thread", "clicked", self.stop_running_worker),
                # 1. dataset tab page
                ("experiments_clustering", "toggled", self.update_metrics),
                ("evaluate_dataset", "clicked", self.run_dataset_evaluation_pipeline),
                ("collect_adsorption_data", "clicked", self.collect_data_from_NIST),
                (
                    "retrieve_guest_properties",
                    "clicked",
                    self.retrieve_guest_properties_from_PUBCHEM,
                ),
                (
                    "retrieve_host_properties",
                    "clicked",
                    self.retrieve_host_properties_from_PUBCHEM,
                ),
                ("build_ML_dataset", "clicked", self.run_dataset_builder),
                # 2. model tab page
                ("start_training", "clicked", self.train_from_scratch),
                ("resume_training", "clicked", self.resume_training_from_checkpoint),
                # model inference and evaluation
                ("checkpoints_list", "currentTextChanged", self.select_checkpoint),
                ("refresh_checkpoints", "clicked", self.load_checkpoints),
                ("get_evaluation_report", "toggled", self.update_metrics),
                ("get_prediction_quality", "toggled", self.update_metrics),
                ("model_evaluation", "clicked", self.run_model_evaluation_pipeline),
                ("checkpoints_summary", "clicked", self.get_checkpoints_summary),
                ("load_data_sources", "clicked", self.update_database_from_sources),
                ("predict_adsorption", "clicked", self.predict_adsorption_isotherms),
                # 3. viewer tab page
                ("train_metrics_view", "toggled", self.update_graphics_view),
                ("previous_image", "clicked", self.show_previous_figure),
                ("next_image", "clicked", self.show_next_figure),
                ("clear_images", "clicked", self.clear_figures),
            ]
        )

        self.auto_connect_settings()

        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self.set_graphics()

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self.widgets[name]
        except (AttributeError, KeyError) as e:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from e

    # [SHOW WINDOW]
    ###########################################################################
    def show(self) -> None:
        self.main_win.show()

    # [HELPERS]
    ###########################################################################
    def connect_update_setting(
        self, widget: Any, signal_name: str, config_key: str, getter: Any | None = None
    ):
        if getter is None:
            if isinstance(widget, (QCheckBox, QRadioButton)):
                getter = widget.isChecked
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                getter = widget.value
            elif isinstance(widget, QComboBox):
                getter = widget.currentText

        signal = getattr(widget, signal_name)
        signal.connect(partial(self.update_single_setting, config_key, getter))

    # -------------------------------------------------------------------------
    def update_single_setting(self, config_key: str, getter: Any, *args) -> None:
        value = getter()
        self.config_manager.update_value(config_key, value)

    # -------------------------------------------------------------------------
    def auto_connect_settings(self) -> None:
        connections = [
            # 1. dataset tab page
            # dataset fetching group
            ("guest_fraction", "valueChanged", "guest_fraction"),
            ("host_fraction", "valueChanged", "host_fraction"),
            ("experiments_fraction", "valueChanged", "experiments_fraction"),
            ("parallel_tasks", "valueChanged", "parallel_tasks"),
            # dataset evaluation and processing group
            ("seed", "valueChanged", "seed"),
            ("sample_size", "valueChanged", "sample_size"),
            ("min_measurements", "valueChanged", "min_measurements"),
            ("max_measurements", "valueChanged", "max_measurements"),
            ("SMILE_sequence_size", "valueChanged", "SMILE_sequence_size"),
            ("max_pressure", "valueChanged", "max_pressure"),
            ("max_uptake", "valueChanged", "max_pressure"),
            ("split_seed", "valueChanged", "split_seed"),
            # 2. model tab page
            # # dataset settings group
            ("use_shuffle", "toggled", "shuffle_dataset"),
            ("shuffle_size", "valueChanged", "shuffle_size"),
            # device settings group
            ("use_device_GPU", "toggled", "use_device_GPU"),
            ("device_ID", "valueChanged", "device_id"),
            ("num_workers", "valueChanged", "num_workers"),
            # training settings group
            ("use_mixed_precision", "toggled", "mixed_precision"),
            ("use_JIT_compiler", "toggled", "use_jit_compiler"),
            ("jit_backend", "currentTextChanged", "jit_backend"),
            ("use_tensorboard", "toggled", "use_tensorboard"),
            ("real_time_history_callback", "toggled", "real_time_history_callback"),
            ("save_checkpoints", "toggled", "save_checkpoints"),
            ("checkpoints_frequency", "valueChanged", "checkpoints_frequency"),
            ("epochs", "valueChanged", "epochs"),
            ("batch_size", "valueChanged", "batch_size"),
            ("train_seed", "valueChanged", "train_seed"),
            # RL scheduler settings group
            ("LR_scheduler", "toggled", "use_LR_scheduler"),
            ("initial_LR", "valueChanged", "initial_LR"),
            ("target_LR", "valueChanged", "target_LR"),
            ("constant_steps", "valueChanged", "constant_steps"),
            ("decay_steps", "valueChanged", "decay_steps"),
            # model settings group
            ("selected_model", "currentTextChanged", "selected_model"),
            ("dropout_rate", "valueChanged", "dropout_rate"),
            ("num_attention_heads", "valueChanged", "num_attention_heads"),
            ("num_encoders", "valueChanged", "num_encoders"),
            ("molecular_embedding_size", "valueChanged", "molecular_embedding_size"),
            # session settings group
            ("additional_epochs", "valueChanged", "additional_epochs"),
            # model inference and evaluation
            ("inference_batch_size", "valueChanged", "inference_batch_size"),
            ("num_evaluation_samples", "valueChanged", "num_evaluation_samples"),
        ]

        self.data_metrics = [("experiments_clustering", self.experiments_clustering)]
        self.model_metrics = [
            ("evaluation_report", self.get_evaluation_report),
            ("prediction_quality", self.get_prediction_quality),
        ]

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

    # -------------------------------------------------------------------------
    def set_states(self) -> None:
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")
        self.progress_bar.setValue(0) if self.progress_bar else None

    # -------------------------------------------------------------------------
    def get_current_pixmaps_key(self) -> tuple[list[Any], str | None]:
        for radio, idx_key in self.pixmap_sources.items():
            if radio and radio.isChecked():
                pixmaps = self.pixmaps.setdefault(idx_key, [])
                self.pixmap_stream_index.setdefault(idx_key, {})
                self.current_fig.setdefault(idx_key, 0)
                return pixmaps, idx_key
        return [], None

    # -------------------------------------------------------------------------
    def set_graphics(self) -> None:
        view = self.main_win.findChild(QGraphicsView, "canvas")
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene) if view else None
        for hint in (
            QPainter.RenderHint.Antialiasing,
            QPainter.RenderHint.SmoothPixmapTransform,
            QPainter.RenderHint.TextAntialiasing,
        ):
            if view:
                view.setRenderHint(hint, True)

        self.graphics = {"view": view, "scene": scene, "pixmap_item": pixmap_item}
        view_keys = "train_metrics"
        self.pixmaps = {k: [] for k in view_keys}
        self.current_fig = {k: 0 for k in view_keys}
        self.pixmap_stream_index = {k: {} for k in view_keys}
        self.pixmap_sources = {
            self.train_metrics_view: "train_metrics",
        }

    # -------------------------------------------------------------------------
    @Slot(object)
    def on_worker_progress(self, payload: Any) -> None:
        try:
            if isinstance(payload, (int, float)):
                if self.progress_bar:
                    self.progress_bar.setValue(int(payload))
                return

            if not isinstance(payload, dict) or payload.get("kind") != "render":
                return

            data = payload.get("data")
            if not data:
                return

            source = payload.get("source", "train_metrics")
            pixmap: QPixmap | None = None

            if isinstance(data, (bytes, bytearray)):
                pixmap = QPixmap()
                if not pixmap.loadFromData(bytes(data)):
                    return
            elif isinstance(data, QPixmap):
                pixmap = data
            elif isinstance(data, str):
                try:
                    pixmap = self.graphic_handler.load_image_as_pixmap(data)
                except Exception:
                    return
            else:
                return

            pixmap_list = self.pixmaps.setdefault(source, [])
            index_map = self.pixmap_stream_index.setdefault(source, {})
            self.current_fig.setdefault(source, 0)

            stream = payload.get("stream")
            if stream:
                idx = index_map.get(stream)
                if idx is not None and idx < len(pixmap_list):
                    pixmap_list[idx] = pixmap
                else:
                    idx = len(pixmap_list)
                    pixmap_list.append(pixmap)
                    index_map[stream] = idx
                    if len(pixmap_list) == 1:
                        self.current_fig[source] = idx
                if self.current_fig.get(source, 0) == idx:
                    self.update_graphics_view()
                return

            pixmap_list.append(pixmap)
            self.current_fig[source] = len(pixmap_list) - 1
            self.update_graphics_view()
        except Exception:
            logger.debug("Unable to handle progress payload", exc_info=True)

    # -------------------------------------------------------------------------
    def reset_train_metrics_stream(self) -> None:
        for key in "train_metrics":
            if key not in self.pixmaps:
                continue
            self.pixmaps[key].clear()
            self.current_fig[key] = 0
            self.pixmap_stream_index[key] = {}
            current_radio = getattr(self, f"{key}view", None)
            if current_radio and current_radio.isChecked():
                self.update_graphics_view()

    # -------------------------------------------------------------------------
    def connect_button(self, button_name: str, slot: Any) -> None:
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) if button else None

    # -------------------------------------------------------------------------
    def connect_combo_box(self, combo_name: str, slot: Any) -> None:
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot) if combo else None

    # -------------------------------------------------------------------------
    def start_thread_worker(
        self,
        worker: ThreadWorker,
        on_finished: Callable,
        on_error: Callable,
        on_interrupted: Callable,
        update_progress: bool = True,
    ) -> None:
        if update_progress and self.progress_bar:
            self.progress_bar.setValue(0) if self.progress_bar else None
        worker.signals.progress.connect(self.on_worker_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)

    # -------------------------------------------------------------------------
    def start_process_worker(
        self,
        worker: ProcessWorker,
        on_finished: Callable,
        on_error: Callable,
        on_interrupted: Callable,
        update_progress: bool = True,
    ) -> None:
        if update_progress and self.progress_bar:
            self.progress_bar.setValue(0) if self.progress_bar else None
        worker.signals.progress.connect(self.on_worker_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        worker.signals.interrupted.connect(on_interrupted)
        # Polling for results from the process queue
        self.process_worker_timer = QTimer()
        self.process_worker_timer.setInterval(100)
        self.process_worker_timer.timeout.connect(worker.poll)
        worker.timer = self.process_worker_timer
        self.process_worker_timer.start()

        worker.start()

    # -------------------------------------------------------------------------
    def send_message(self, message: str) -> None:
        self.main_win.statusBar().showMessage(message)

    # [SETUP]
    ###########################################################################
    def setup_configuration(self, widget_defs: Any) -> None:
        for cls, name, attr in widget_defs:
            w = self.main_win.findChild(cls, name)
            setattr(self, attr, w)
            self.widgets[attr] = w

    # -------------------------------------------------------------------------
    def connect_signals(self, connections: Any) -> None:
        for attr, signal, slot in connections:
            widget = self.widgets[attr]
            getattr(widget, signal).connect(slot)

    # -------------------------------------------------------------------------
    def set_widgets_from_configuration(self) -> None:
        cfg = self.config_manager.get_configuration()
        for attr, widget in self.widgets.items():
            if attr not in cfg:
                continue
            v = cfg[attr]

            if hasattr(widget, "setChecked") and isinstance(v, bool):
                widget.setChecked(v)
            elif hasattr(widget, "setValue") and isinstance(v, (int, float)):
                widget.setValue(v)
            elif hasattr(widget, "setPlainText") and isinstance(v, str):
                widget.setPlainText(v)
            elif hasattr(widget, "setText") and isinstance(v, str):
                widget.setText(v)
            elif isinstance(widget, QComboBox):
                if isinstance(v, str):
                    idx = widget.findText(v)
                    if idx != -1:
                        widget.setCurrentIndex(idx)
                    elif widget.isEditable():
                        widget.setEditText(v)
                elif isinstance(v, int) and 0 <= v < widget.count():
                    widget.setCurrentIndex(v)

    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    # -------------------------------------------------------------------------
    @Slot()
    def stop_running_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.send_message("Interrupt requested. Waiting for threads to stop...")

    # -------------------------------------------------------------------------
    @Slot()
    def load_checkpoints(self) -> None:
        checkpoints = self.model_handler.get_available_checkpoints()
        self.checkpoints_list.clear()
        if checkpoints:
            self.checkpoints_list.addItems(checkpoints)
            self.selected_checkpoint = checkpoints[0]
            self.checkpoints_list.setCurrentText(checkpoints[0])
        else:
            self.selected_checkpoint = None
            logger.warning("No checkpoints available")

    # -------------------------------------------------------------------------
    @Slot()
    def select_checkpoint(self, name: str) -> None:
        self.selected_checkpoint = name if name else None

    # -------------------------------------------------------------------------
    @Slot()
    def update_metrics(self) -> None:
        self.selected_metrics["dataset"] = [
            name for name, box in self.data_metrics if box and box.isChecked()
        ]
        self.selected_metrics["model"] = [
            name for name, box in self.model_metrics if box and box.isChecked()
        ]

    # -------------------------------------------------------------------------
    # [ACTIONS]
    # -------------------------------------------------------------------------
    @Slot()
    def save_configuration(self) -> None:
        dialog = SaveConfigDialog(self.main_win)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = dialog.get_name()
            name = "default_config" if not name else name
            self.config_manager.save_configuration_to_json(name)
            self.send_message(f"Configuration [{name}] has been saved")

    # -------------------------------------------------------------------------
    @Slot()
    def load_configuration(self) -> None:
        dialog = LoadConfigDialog(self.main_win)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = dialog.get_selected_config()
            if name:
                self.config_manager.load_configuration_from_json(name)
                self.set_widgets_from_configuration()
                self.send_message(f"Loaded configuration [{name}]")

    # -------------------------------------------------------------------------
    @Slot()
    def export_all_data(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self.main_win, "Select export directory"
        )
        if not directory:
            message = "Export cancelled"
            logger.info(message)
            self.send_message(message)
            return

        database.export_all_tables_as_csv(directory)
        message = f"All data from database has been exported to {directory}"
        logger.info(message)
        self.send_message(message)

    # -------------------------------------------------------------------------
    @Slot()
    def delete_all_data(self) -> None:
        database.delete_all_data()
        message = "All data from database has been deleted"
        logger.info(message)
        self.send_message(message)

    # -------------------------------------------------------------------------
    # [GRAPHICS]
    # -------------------------------------------------------------------------
    @Slot()
    def update_graphics_view(self) -> None:
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            self.graphics["pixmap_item"].setPixmap(QPixmap())
            self.graphics["scene"].setSceneRect(0, 0, 0, 0)
            return

        idx = self.current_fig.get(idx_key, 0)
        idx = min(idx, len(pixmaps) - 1)
        raw = pixmaps[idx]

        qpixmap = QPixmap(raw) if isinstance(raw, str) else raw
        view = self.graphics["view"]
        pixmap_item = self.graphics["pixmap_item"]
        scene = self.graphics["scene"]
        view_size = view.viewport().size()
        scaled = qpixmap.scaled(
            view_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())

    # -------------------------------------------------------------------------
    @Slot()
    def show_previous_figure(self) -> None:
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self.update_graphics_view()

    # -------------------------------------------------------------------------
    @Slot()
    def show_next_figure(self) -> None:
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self.update_graphics_view()

    # -------------------------------------------------------------------------
    @Slot()
    def clear_figures(self) -> None:
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        self.pixmaps[idx_key].clear() if idx_key else None
        self.current_fig[idx_key] = 0
        if idx_key in self.pixmap_stream_index:
            self.pixmap_stream_index[idx_key] = {}
        self.update_graphics_view()
        self.graphics["pixmap_item"].setPixmap(QPixmap())
        self.graphics["scene"].setSceneRect(0, 0, 0, 0)
        self.graphics["view"].viewport().update()

    # -------------------------------------------------------------------------
    # [DATASET TAB]
    # -------------------------------------------------------------------------
    @Slot()
    def collect_data_from_NIST(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.dataset_handler = DatasetEvents(self.configuration)
        # send message to status bar
        self.send_message(
            "Collecting adsorption isotherms and materials data from NIST-A database..."
        )

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.dataset_handler.run_data_collection_pipeline)

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_data_success,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def retrieve_guest_properties_from_PUBCHEM(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.dataset_handler = DatasetEvents(self.configuration)
        # send message to status bar
        self.send_message("Retrieving molecular properties from Pubchem API...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(
            self.dataset_handler.run_chemical_properties_pipeline, target="guest"
        )

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_data_success,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def retrieve_host_properties_from_PUBCHEM(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.dataset_handler = DatasetEvents(self.configuration)
        # send message to status bar
        self.send_message("Retrieving molecular properties from Pubchem API...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(
            self.dataset_handler.run_chemical_properties_pipeline, target="host"
        )

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_data_success,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def run_dataset_evaluation_pipeline(self) -> None:
        if not self.selected_metrics["dataset"]:
            return

        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.validation_handler = ValidationEvents(self.configuration)
        # send message to status bar
        self.send_message("Evaluate adsorption isotherms and materials dataset...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics["dataset"],
        )

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def run_dataset_builder(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.dataset_handler = DatasetEvents(self.configuration)
        # send message to status bar
        self.send_message("Building preprocessed machine learning dataset...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.dataset_handler.run_dataset_builder)

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_dataset_processing_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    # [TRAINING TAB]
    # -------------------------------------------------------------------------
    @Slot()
    def train_from_scratch(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.model_handler = ModelEvents(self.configuration)
        self.reset_train_metrics_stream()

        # send message to status bar
        self.send_message("Training SCADS using a new model instance...")
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(self.model_handler.run_training_pipeline)

        # start worker and inject signals
        self.start_process_worker(
            self.worker,
            on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        if not self.selected_checkpoint:
            logger.warning("No checkpoint selected for resuming training")
            return

        self.configuration = self.config_manager.get_configuration()
        self.model_handler = ModelEvents(self.configuration)
        self.reset_train_metrics_stream()

        # send message to status bar
        self.send_message(
            f"Resume training from checkpoint {self.selected_checkpoint}"
        )
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.resume_training_pipeline, self.selected_checkpoint
        )

        # start worker and inject signals
        self.start_process_worker(
            self.worker,
            on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    # [MODEL EVALUATION TAB]
    # -------------------------------------------------------------------------
    @Slot()
    def run_model_evaluation_pipeline(self) -> None:
        if not self.selected_metrics["model"]:
            return

        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.validation_handler = ValidationEvents(self.configuration)

        # send message to status bar
        self.send_message(f"Evaluating {self.selected_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics["model"],
            self.selected_checkpoint,
        )

        # start worker and inject signals
        self.start_process_worker(
            self.worker,
            on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def get_checkpoints_summary(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.validation_handler = ValidationEvents(self.configuration)
        # send message to status bar
        self.send_message("Generating checkpoints summary...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.validation_handler.get_checkpoints_summary)

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    # [INFERENCE TAB]
    # -------------------------------------------------------------------------
    @Slot()
    def update_database_from_sources(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        # send message to status bar
        self.send_message("Updating database with source data...")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(database.update_database_from_sources)

        # start worker and inject signals
        self.start_thread_worker(
            self.worker,
            on_finished=self.on_database_uploading_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    # -------------------------------------------------------------------------
    @Slot()
    def predict_adsorption_isotherms(self) -> None:
        if self.worker:
            message = (
                "A task is currently running, wait for it to finish and then try again"
            )
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return

        self.configuration = self.config_manager.get_configuration()
        self.model_handler = ModelEvents(self.configuration)

        # send message to status bar
        self.send_message(f"Encoding images with {self.selected_checkpoint}")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.run_inference_pipeline, self.selected_checkpoint
        )

        # start worker and inject signals
        self.start_process_worker(
            self.worker,
            on_finished=self.on_inference_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted,
        )

    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################
    def on_database_uploading_finished(self, source_data: pd.DataFrame | None) -> None:
        if source_data is None:
            message = "No source data found in resources"
            return

        message = (
            f"Database updated with current source data ({len(source_data)}) records"
        )
        self.send_message(message)
        QMessageBox.information(self.main_win, "Database successfully updated", message)
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_data_success(self, session: dict[str, Any]) -> None:
        self.send_message("Data has been collected from NIST-A database")
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_dataset_processing_finished(self, session: dict[str, Any]) -> None:
        self.send_message("Dataset has been built successfully")
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_dataset_evaluation_finished(self, plots: list[Any]) -> None:
        self.send_message("Figures have been generated")
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_train_finished(self, session: dict[str, Any]) -> None:
        self.send_message("Training session is over. Model has been saved")
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots: list[Figure]) -> None:
        self.send_message(f"Model {self.selected_checkpoint} has been evaluated")
        self.worker = self.worker.cleanup() if self.worker else None

    # -------------------------------------------------------------------------
    def on_inference_finished(self, session: dict[str, Any]) -> None:
        self.send_message("Inference call has been terminated")
        self.worker = self.worker.cleanup() if self.worker else None

    ###########################################################################
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################
    def on_error(self, err_tb: tuple[str, str]) -> None:
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        message = "An error occurred during the operation. Check the logs for details."
        QMessageBox.critical(self.main_win, "Something went wrong!", message)
        self.progress_bar.setValue(0) if self.progress_bar else None
        self.worker = self.worker.cleanup() if self.worker else None

    ###########################################################################
    # [INTERRUPTION HANDLERS]
    ###########################################################################
    def on_task_interrupted(self) -> None:
        self.progress_bar.setValue(0) if self.progress_bar else None
        self.send_message("Current task has been interrupted by user")
        logger.warning("Current task has been interrupted by user")
        self.worker = self.worker.cleanup() if self.worker else None
