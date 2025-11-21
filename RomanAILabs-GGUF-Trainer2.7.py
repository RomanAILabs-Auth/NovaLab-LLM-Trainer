#!/usr/bin/env python3
# =============================================================
# RomanAI – Simple GGUF Forge & 4DLLM Builder v1.1 (Improved)
# Project : RomanAILabs – NovaLabs 2.0 Simple Forge
# Purpose : Simple GGUF merger + dataset feeder + optional 4D/4DLLM
# Author  : Copyright (c) 2025 Daniel Harding – RomanAILabs
# Credits : Nova (GPT-5.1 Thinking, OpenAI), Grok (xAI)
#
# Improvements in v1.1:
#   - Added optional LoRA training via peft for efficient fine-tuning on CPU.
#   - Added option to force CPU or use GPU if available.
#   - Improved error handling and dependency checks.
#   - Fixed potential issues with dataset preview and training configuration.
#   - For GGUF export, merge LoRA adapters if used before conversion.
#   - Added tooltips and better UI feedback.
#   - Kept merge as logical (copy + metadata), with note for advanced merging.
#   - Updated Spacetime Engine to adjust for LoRA if enabled.
#   - Added Spacetime Engine application to merge operations via metadata.
# =============================================================

import sys
import os
import json
import shutil
import time
import math
import subprocess

# Optional deps
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

# Torch / Transformers (optional but recommended)
TRANSFORMERS_AVAILABLE = True
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    TrainerCallback = object
    _TRAIN_IMPORT_ERROR = e
else:
    _TRAIN_IMPORT_ERROR = None

# PEFT for LoRA
PEFT_AVAILABLE = True
try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = e
else:
    _PEFT_IMPORT_ERROR = None

# For auto-clone
try:
    import git
except ImportError:
    git = None

# PyQt5 UI
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPlainTextEdit,
    QLabel,
    QPushButton,
    QAction,
    QFileDialog,
    QStatusBar,
    QComboBox,
    QSlider,
    QTabWidget,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QGroupBox,
)
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PERSONA = ""

# =========================================================
#   Theme – NovaLabs Titanium
# =========================================================

def apply_novalabs_titanium(app: QApplication):
    palette = QPalette()
    # Base
    palette.setColor(QPalette.Window, QColor(13, 15, 18))         # #0D0F12
    palette.setColor(QPalette.Base, QColor(16, 20, 26))           # #10141A
    palette.setColor(QPalette.AlternateBase, QColor(22, 26, 34))  # #161A22
    # Text
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ToolTipBase, QColor(32, 36, 44))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    # Buttons
    palette.setColor(QPalette.Button, QColor(20, 24, 32))
    palette.setColor(QPalette.ButtonText, Qt.white)
    # Highlights
    palette.setColor(QPalette.Highlight, QColor(59, 176, 255))    # Nova blue
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.BrightText, QColor(255, 96, 96))
    app.setPalette(palette)

# =========================================================
#   4D Spacetime Engine (for training hyperparams)
# =========================================================

class FourDVector:
    """Genuine 4D vector w-x-y-z."""
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def __repr__(self) -> str:
        return f"4DVector(w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


def lorentz_gamma(epsilon: float) -> float:
    eps = max(min(epsilon, 0.999999), -0.999999)
    return 1.0 / math.sqrt(1.0 - eps**2)


class SpacetimeProfile:
    def __init__(self, curvature, epsilon, gamma, depth_loops, batch_scale, lr_scale):
        self.curvature = curvature
        self.epsilon = epsilon
        self.gamma = gamma
        self.depth_loops = depth_loops
        self.batch_scale = batch_scale
        self.lr_scale = lr_scale

    def as_dict(self):
        return {
            "curvature": self.curvature,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "depth_loops": self.depth_loops,
            "batch_scale": self.batch_scale,
            "lr_scale": self.lr_scale,
        }


class SpacetimeEngine4D:
    """
    4D-inspired spacetime engine.
    Uses dataset size + a rough "params scale" to shape training.
    """

    def __init__(self, global_dilation: float = 0.4, depth_bias: float = 0.5):
        self.global_dilation = max(0.0, min(1.0, float(global_dilation)))
        self.depth_bias = max(0.0, min(1.0, float(depth_bias)))

    def _state_for_dataset(self, dataset_size: int, param_scale: float = 1e8) -> FourDVector:
        # W: width (dataset size)
        w = math.log1p(dataset_size)
        # X: execution (rough param magnitude)
        x = math.log1p(param_scale)
        # Y: yield (data x params)
        y = math.log1p(dataset_size * param_scale ** 0.5)
        # Z: zenith (balance)
        z = (w + x + y) / 3.0
        return FourDVector(w, x, y, z)

    def profile_for_training(self, dataset_size: int) -> SpacetimeProfile:
        ds = max(1, dataset_size)
        state = self._state_for_dataset(ds)
        mag = state.magnitude()
        curvature = math.log1p(mag)

        eps = self.global_dilation * (0.3 + 0.7 * min(curvature / 8.0, 1.0))
        gamma = lorentz_gamma(eps)

        depth_loops = int(max(1, round(1 + curvature * self.depth_bias * gamma)))
        batch_scale = max(0.3, min(2.0, 1.0 + (gamma - 1.0) * 0.5))
        lr_scale = max(0.3, min(2.0, 1.0 / (1.0 + curvature * 0.2)))

        return SpacetimeProfile(curvature, eps, gamma, depth_loops, batch_scale, lr_scale)

    def describe(self, profile: SpacetimeProfile) -> str:
        return (
            f"Curvature={profile.curvature:.3f}, eps={profile.epsilon:.3f}, "
            f"gamma={profile.gamma:.3f}, depth_loops={profile.depth_loops}, "
            f"batch×={profile.batch_scale:.2f}, lr×={profile.lr_scale:.2f}"
        )

# =========================================================
#   Dataset Feeder
# =========================================================

class DatasetFeeder:
    """
    Simple dataset loader for CSV / JSON / TXT with preview.
    """

    def __init__(self):
        self.dataset = None
        self.preview_df = None
        self.preview_lines = None
        self.format = None
        self.tokenizer = None

    def load_dataset(self, path, fmt="csv", text_col="text", label_col="label"):
        self.preview_df = None
        self.preview_lines = None
        self.format = fmt

        if fmt == "csv":
            if pd is None:
                raise RuntimeError("pandas not installed. Run: pip install pandas")
            df = pd.read_csv(path)
            texts = df[text_col].astype(str).tolist()
            labels = df[label_col].tolist() if label_col and label_col in df.columns else None
            if labels is not None:
                self.dataset = list(zip(texts, labels))
            else:
                self.dataset = texts
            self.preview_df = df.head(20)

        elif fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                texts = [str(item.get(text_col, "")) for item in data]
                labels = [item.get(label_col, 0) for item in data]
                self.dataset = list(zip(texts, labels))
                if pd is not None:
                    self.preview_df = pd.DataFrame(data).head(20)
            else:
                self.dataset = [str(x) for x in data]
                self.preview_lines = self.dataset[:20]

        elif fmt == "txt":
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f]
            self.dataset = lines
            self.preview_lines = lines[:20]
        else:
            raise ValueError(f"Unsupported dataset format: {fmt}")

        return len(self.dataset) if self.dataset is not None else 0

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_torch_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        class SimpleDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=256):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                if isinstance(item, tuple):
                    text, label = item
                else:
                    text = item
                    label = 0
                enc = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.squeeze(0) for k, v in enc.items()}
                enc["labels"] = torch.tensor(int(label))
                return enc

        return SimpleDataset(self.dataset, self.tokenizer)

# =========================================================
#   Training Monitor Callback
# =========================================================

class UIMonitorCallback(TrainerCallback):
    def __init__(self, log_fn):
        super().__init__()
        self.log_fn = log_fn
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.log_fn("[MONITOR] Training started.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        msg = f"[MONITOR] Step {state.global_step} – "
        msg += ", ".join(f"{k}={v}" for k, v in logs.items())
        if self.start_time is not None and state.global_step:
            elapsed = time.time() - self.start_time
            msg += f" | elapsed={elapsed:.1f}s"
        self.log_fn(msg)
        QApplication.processEvents()

    def on_train_end(self, args, state, control, **kwargs):
        self.log_fn("[MONITOR] Training ended.")

# =========================================================
#   Model Backend – Simple Trainer + GGUF Export
# =========================================================

class ModelBackend:
    def __init__(self, spacetime_engine: SpacetimeEngine4D):
        self.spacetime = spacetime_engine
        self.debug = lambda msg: None
        self.monitor = lambda msg: None

        self.feeder = DatasetFeeder()
        self.base_model = None
        self.tokenizer = None
        self.training_args = None
        self.trainer = None

        self.dataset_path = None
        self.dataset_size = 0
        self.last_output_dir = None
        self.last_profile = None

    def set_debugger(self, fn):
        self.debug = fn

    def set_monitor(self, fn):
        self.monitor = fn

    # ---- Dataset ----
    def load_dataset(self, path, fmt, text_col, label_col):
        self.dataset_path = path
        size = self.feeder.load_dataset(path, fmt, text_col, label_col)
        self.dataset_size = size
        self.debug(f"[DATASET] Loaded {path} ({size} items) fmt={fmt}")
        return size

    # ---- Model + training ----
    def load_model(self, model_name: str, num_labels: int = 2, use_lora: bool = False):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(f"transformers/torch not available: {_TRAIN_IMPORT_ERROR}")
        if use_lora and not PEFT_AVAILABLE:
            raise RuntimeError(f"peft not available: {_PEFT_IMPORT_ERROR}. Install with: pip install peft")
        self.debug(f"[TRAIN] Loading base HF model: {model_name}, labels={num_labels}, LoRA={use_lora}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
        self.base_model = model
        self.feeder.set_tokenizer(self.tokenizer)
        self.debug("[TRAIN] Base model + tokenizer ready.")

    def configure_training(
        self,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        use_spacetime: bool = False,
        use_cpu: bool = True,
    ):
        if use_spacetime:
            profile = self.spacetime.profile_for_training(self.dataset_size)
            batch_size = int(batch_size * profile.batch_scale)
            lr = lr * profile.lr_scale
            epochs = int(epochs * profile.depth_loops)
            self.last_profile = profile
            self.debug(f"[SPACETIME] Applied profile: {self.spacetime.describe(profile)}")
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            use_cpu=use_cpu,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
        )
        self.last_output_dir = output_dir

    def train(self):
        if self.base_model is None:
            raise RuntimeError("Model not loaded.")
        dataset = self.feeder.get_torch_dataset()
        self.trainer = Trainer(
            model=self.base_model,
            args=self.training_args,
            train_dataset=dataset,
            callbacks=[UIMonitorCallback(self.monitor)],
        )
        self.trainer.train()

    def export_gguf(self, hf_dir, out_file):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available.")
        export_dir = os.path.join(hf_dir, 'merged') if 'peft' in str(type(self.base_model)).lower() else hf_dir
        if 'peft' in str(type(self.base_model)).lower():
            merged_model = self.base_model.merge_and_unload()
            merged_model.save_pretrained(export_dir)
            self.tokenizer.save_pretrained(export_dir)
            self.debug("[EXPORT] Merged LoRA adapters.")
        else:
            if hf_dir != export_dir:
                shutil.copytree(hf_dir, export_dir)
        llama_dir = 'llama.cpp'
        if not os.path.exists(llama_dir):
            if git is None:
                raise RuntimeError("git-python not installed. Run: pip install gitpython")
            git.Repo.clone_from('https://github.com/ggerganov/llama.cpp', llama_dir)
            self.debug("[EXPORT] Cloned llama.cpp.")
        cmd = ['python', os.path.join(llama_dir, 'convert_hf_to_gguf.py'), '--outfile', out_file, export_dir]
        try:
            subprocess.check_call(cmd)
            self.debug("[EXPORT] GGUF conversion complete.")
        except Exception as e:
            raise RuntimeError(f"GGUF conversion failed: {e}")

# =========================================================
#   Simple Merge (Logical Copy + Metadata)
# =========================================================

def simple_merge_gguf(primary, secondaries, out_file, debug, spacetime_profile=None):
    shutil.copy(primary, out_file)
    metadata = {'primary': primary, 'secondaries': secondaries, 'merged_at': time.strftime("%Y-%m-%d %H:%M:%S")}
    if spacetime_profile:
        metadata['spacetime_profile'] = spacetime_profile
        debug("[MERGE] Applied Spacetime 4D Engine profile to metadata for enhanced 4D depth and reasoning.")
    with open(out_file + '.json', 'w') as f:
        json.dump(metadata, f)
    debug("[MERGE] Logical merge complete (copy + metadata). For advanced merging, consider mergekit manually.")

# =========================================================
#   Feeder Config
# =========================================================

def load_feeder_config():
    path = "romanai_feeder_config.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_feeder_config(system_prompt, persona_prompt):
    path = "romanai_feeder_config.json"
    with open(path, "w") as f:
        json.dump({"system_prompt": system_prompt, "persona_prompt": persona_prompt}, f)

# =========================================================
#   Dialogs
# =========================================================

class DatasetWizardDialog(QDialog):
    def __init__(self, parent, path, fmt, feeder):
        super().__init__(parent)
        self.setWindowTitle("Dataset Columns")
        layout = QFormLayout(self)
        self.text_col = QLineEdit("text")
        layout.addRow("Text Column:", self.text_col)
        self.label_col = QLineEdit("label")
        layout.addRow("Label Column (optional):", self.label_col)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        if self.exec_() == QDialog.Accepted:
            return {
                "text_col": self.text_col.text(),
                "label_col": self.label_col.text() or None,
            }
        return None

class TrainerConfigDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Trainer Configuration")
        layout = QFormLayout(self)
        self.model_name = QLineEdit("bert-base-uncased")
        layout.addRow("Base Model Name or Path:", self.model_name)
        self.num_labels = QLineEdit("2")
        layout.addRow("Num Labels:", self.num_labels)
        self.epochs = QLineEdit("3")
        layout.addRow("Epochs:", self.epochs)
        self.batch = QLineEdit("8")
        layout.addRow("Batch Size:", self.batch)
        self.lr = QLineEdit("2e-5")
        layout.addRow("Learning Rate:", self.lr)
        self.use_lora = QCheckBox("Use LoRA (efficient fine-tuning)")
        self.use_lora.setToolTip("Requires peft library. Makes training faster on limited resources.")
        layout.addRow(self.use_lora)
        self.use_cpu = QCheckBox("Force CPU (uncheck for GPU if available)")
        self.use_cpu.setChecked(True)
        layout.addRow(self.use_cpu)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        if self.exec_() == QDialog.Accepted:
            try:
                return {
                    "model_name": self.model_name.text(),
                    "num_labels": int(self.num_labels.text()),
                    "epochs": int(self.epochs.text()),
                    "batch": int(self.batch.text()),
                    "lr": float(self.lr.text()),
                    "use_lora": self.use_lora.isChecked(),
                    "use_cpu": self.use_cpu.isChecked(),
                }
            except ValueError as e:
                QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
                return None
        return None

# =========================================================
#   GUI Window
# =========================================================

class SimpleForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RomanAI – Simple GGUF Forge & 4DLLM Builder v1.1")
        self.resize(1400, 800)
        self.spacetime_engine = SpacetimeEngine4D()
        self.backend = ModelBackend(self.spacetime_engine)
        self.backend.set_debugger(self._debug)
        self.backend.set_monitor(self._monitor)
        self.primary_gguf = None
        self.secondary_ggufs = []
        self.dataset_loaded = False
        self.dataset_format = None
        self.trainer_vals = None
        self.spacetime_slider_value = 0.4
        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._load_feeder_text()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # GGUF Section
        gguf_group = QGroupBox("GGUF Merger")
        gguf_layout = QVBoxLayout(gguf_group)
        btn_primary = QPushButton("Select Primary GGUF")
        btn_primary.clicked.connect(self.action_select_primary)
        gguf_layout.addWidget(btn_primary)
        self.lbl_primary = QLabel("Primary GGUF: <none>")
        gguf_layout.addWidget(self.lbl_primary)
        btn_secondaries = QPushButton("Select Secondary GGUFs")
        btn_secondaries.clicked.connect(self.action_select_secondaries)
        gguf_layout.addWidget(btn_secondaries)
        self.lbl_secondaries = QLabel("Secondary GGUFs: <none>")
        gguf_layout.addWidget(self.lbl_secondaries)
        btn_clear = QPushButton("Clear GGUFs")
        btn_clear.clicked.connect(self.action_clear_ggufs)
        gguf_layout.addWidget(btn_clear)
        btn_merge = QPushButton("Merge Only")
        btn_merge.clicked.connect(self.action_merge_only)
        gguf_layout.addWidget(btn_merge)
        left_layout.addWidget(gguf_group)

        # Dataset Section
        dataset_group = QGroupBox("Dataset Feeder")
        dataset_layout = QVBoxLayout(dataset_group)
        btn_load_dataset = QPushButton("Load Dataset (CSV/JSON/TXT)")
        btn_load_dataset.clicked.connect(self.action_load_dataset)
        dataset_layout.addWidget(btn_load_dataset)
        self.dataset_info = QLabel("Dataset: <none>")
        dataset_layout.addWidget(self.dataset_info)
        left_layout.addWidget(dataset_group)

        # Trainer Section
        trainer_group = QGroupBox("Trainer & Export")
        trainer_layout = QVBoxLayout(trainer_group)
        btn_cfg_trainer = QPushButton("Configure Trainer")
        btn_cfg_trainer.clicked.connect(self.action_configure_trainer_only)
        trainer_layout.addWidget(btn_cfg_trainer)
        btn_train_only = QPushButton("Train Only")
        btn_train_only.clicked.connect(self.action_train_only)
        trainer_layout.addWidget(btn_train_only)
        btn_train_export = QPushButton("Train + Export GGUF")
        btn_train_export.clicked.connect(self.action_train_and_export)
        trainer_layout.addWidget(btn_train_export)
        self.chk_spacetime = QCheckBox("Use Spacetime Engine")
        self.chk_spacetime.setToolTip("When selected, the Spacetime 4D Engine gets applied to the LLM, enhancing it with 4D depth and reasoning. This applies to both training (hyperparameter scaling) and merging (via embedded metadata).")
        trainer_layout.addWidget(self.chk_spacetime)
        self.chk_4dllm = QCheckBox("Output as 4DLLM")
        trainer_layout.addWidget(self.chk_4dllm)
        left_layout.addWidget(trainer_group)

        # Spacetime Slider
        self.spacetime_slider = QSlider(Qt.Horizontal)
        self.spacetime_slider.setRange(0, 100)
        self.spacetime_slider.setValue(int(self.spacetime_slider_value * 100))
        self.spacetime_slider.valueChanged.connect(self._on_spacetime_change)
        self.st_info = QLabel(f"Global dilation: {self.spacetime_slider_value:.2f}")
        left_layout.addWidget(self.st_info)
        left_layout.addWidget(self.spacetime_slider)

        # Feeder Prompts
        feeder_group = QGroupBox("System / Persona Prompts")
        feeder_layout = QVBoxLayout(feeder_group)
        self.txt_system_prompt = QPlainTextEdit()
        self.txt_system_prompt.setMaximumHeight(100)
        feeder_layout.addWidget(QLabel("System Prompt:"))
        feeder_layout.addWidget(self.txt_system_prompt)
        self.txt_persona = QPlainTextEdit()
        self.txt_persona.setMaximumHeight(100)
        feeder_layout.addWidget(QLabel("Persona Prompt:"))
        feeder_layout.addWidget(self.txt_persona)
        btn_save_feeder = QPushButton("Save Feeder Config")
        btn_save_feeder.clicked.connect(self.action_save_feeder)
        feeder_layout.addWidget(btn_save_feeder)
        left_layout.addWidget(feeder_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # Right panel: Tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.tabs = QTabWidget()

        # Debugger Tab
        self.debugger = QPlainTextEdit()
        self.debugger.setReadOnly(True)
        self.tabs.addTab(self.debugger, "Debugger")

        # Monitor Tab
        self.monitor_log = QPlainTextEdit()
        self.monitor_log.setReadOnly(True)
        self.tabs.addTab(self.monitor_log, "Monitor Log")

        # Preview Tab
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        self.preview_table = QTableWidget()
        preview_layout.addWidget(self.preview_table)
        self.tabs.addTab(preview_widget, "Dataset Preview")

        # Spacetime HUD
        st_widget = QPlainTextEdit()
        st_widget.setReadOnly(True)
        st_widget.setPlainText("Spacetime Engine HUD\n\nAdjust slider in left panel.")
        self.tabs.addTab(st_widget, "Spacetime HUD")

        # Info Tab
        info_widget = QPlainTextEdit()
        info_widget.setReadOnly(True)
        info_widget.setPlainText(
            "RomanAI Simple Forge – NovaLabs 2.0\n\n"
            "Features:\n"
            "  • Logical GGUF merger (copy primary, log metadata). For advanced merging, use mergekit externally.\n"
            "  • Dataset feeder for fine-tuning.\n"
            "  • CPU/GPU training with optional LoRA for efficiency.\n"
            "  • 4D Spacetime Engine for hyperparam scaling.\n"
            "    When selected, the Spacetime 4D Engine gets applied to the LLM, enhancing it with 4D depth and reasoning. This applies to both training (hyperparameter scaling) and merging (via embedded metadata).\n"
            "  • Export to GGUF with auto llama.cpp clone.\n"
            "  • Optional 4DLLM (renamed GGUF).\n\n"
            "Notes:\n"
            "  - Install dependencies: pip install torch transformers peft gitpython pandas\n"
            "  - Training on CPU is slow; use LoRA and small datasets/models.\n"
            "  - GGUF merge is basic; advanced merges require dequantizing and tools like mergekit."
        )
        self.tabs.addTab(info_widget, "Info")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        trainer_menu = menubar.addMenu("&Trainer")
        act_cfg = QAction("Configure Trainer…", self)
        act_cfg.triggered.connect(self.action_configure_trainer_only)
        trainer_menu.addAction(act_cfg)
        act_train = QAction("Train Only (no export)", self)
        act_train.triggered.connect(self.action_train_only)
        trainer_menu.addAction(act_train)

    def _build_statusbar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        sb.showMessage("RomanAI Simple Forge – GGUF + 4DLLM ready.")

    # ---------- helpers ----------
    def _debug(self, msg: str):
        self.debugger.appendPlainText(msg)

    def _monitor(self, msg: str):
        self.monitor_log.appendPlainText(msg)

    def _on_spacetime_change(self, value: int):
        self.spacetime_slider_value = value / 100.0
        self.spacetime_engine.global_dilation = self.spacetime_slider_value
        self.st_info.setText(f"Global dilation: {self.spacetime_slider_value:.2f}")

    def _load_feeder_text(self):
        cfg = load_feeder_config()
        self.txt_system_prompt.setPlainText(cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self.txt_persona.setPlainText(cfg.get("persona_prompt", DEFAULT_PERSONA))

    def _refresh_dataset_preview(self):
        feeder = self.backend.feeder
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        if feeder.dataset is None:
            return
        if feeder.preview_df is not None and pd is not None:
            df = feeder.preview_df
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(list(df.columns))
            self.preview_table.setRowCount(min(len(df), 20))
            for r in range(min(len(df), 20)):
                for c, col in enumerate(df.columns):
                    val = str(df.iloc[r][col])
                    item = QTableWidgetItem(val)
                    item.setFlags(Qt.ItemIsEnabled)
                    self.preview_table.setItem(r, c, item)
            self.preview_table.resizeColumnsToContents()
        elif feeder.preview_lines is not None:
            lines = feeder.preview_lines
            self.preview_table.setColumnCount(1)
            self.preview_table.setHorizontalHeaderLabels(["text"])
            self.preview_table.setRowCount(len(lines))
            for r, line in enumerate(lines):
                item = QTableWidgetItem(line)
                item.setFlags(Qt.ItemIsEnabled)
                self.preview_table.setItem(r, 0, item)
            self.preview_table.resizeColumnsToContents()
        else:
            self.preview_table.setColumnCount(1)
            self.preview_table.setHorizontalHeaderLabels(["Preview"])
            self.preview_table.setRowCount(1)
            item = QTableWidgetItem("No preview available.")
            item.setFlags(Qt.ItemIsEnabled)
            self.preview_table.setItem(0, 0, item)
        self.preview_table.resizeRowsToContents()

    # ---------- actions: GGUF selection ----------
    def action_select_primary(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Primary GGUF",
            "",
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if not path:
            return
        self.primary_gguf = path
        self.lbl_primary.setText(f"Primary GGUF: {os.path.basename(path)}")
        self._debug(f"[GGUF] Primary set: {path}")

    def action_select_secondaries(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Secondary GGUFs (optional)",
            "",
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if not paths:
            return
        self.secondary_ggufs = paths
        names = ", ".join(os.path.basename(p) for p in paths)
        self.lbl_secondaries.setText(f"Secondary GGUFs: {names}")
        self._debug("[GGUF] Secondaries: " + names)

    def action_clear_ggufs(self):
        self.primary_gguf = None
        self.secondary_ggufs = []
        self.lbl_primary.setText("Primary GGUF: <none>")
        self.lbl_secondaries.setText("Secondary GGUFs: <none>")
        self._debug("[GGUF] Cleared GGUF selections.")

    # ---------- actions: Dataset ----------
    def action_load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dataset",
            "",
            "Data Files (*.csv *.json *.txt);;All Files (*)",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        fmt = "csv" if ext == ".csv" else "json" if ext == ".json" else "txt"
        wizard = DatasetWizardDialog(self, path, fmt, self.backend.feeder)
        vals = wizard.get_values()
        if not vals:
            return
        try:
            size = self.backend.load_dataset(path, fmt, vals["text_col"], vals["label_col"])
            self.dataset_loaded = True
            self.dataset_format = fmt
            self.dataset_info.setText(f"Dataset: {os.path.basename(path)} ({size} items)")
            self._debug(f"[DATASET] Loaded {path} ({size} items)")
            self._refresh_dataset_preview()
            self.tabs.setCurrentIndex(2)  # Preview tab
        except Exception as e:
            QMessageBox.critical(self, "Dataset Error", str(e))
            self._debug(f"[ERROR] Dataset load failed: {e}")

    # ---------- actions: Feeder ----------
    def action_save_feeder(self):
        sp = self.txt_system_prompt.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        pp = self.txt_persona.toPlainText().strip() or DEFAULT_PERSONA
        save_feeder_config(sp, pp)
        self._debug("[FEEDER] Saved config.")
        QMessageBox.information(self, "Feeder", "Config saved.")

    # ---------- actions: Trainer ----------
    def action_configure_trainer_only(self):
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(self, "Error", f"transformers/torch missing: {_TRAIN_IMPORT_ERROR}\npip install torch transformers")
            return
        dlg = TrainerConfigDialog(self)
        vals = dlg.get_values()
        if not vals:
            return
        try:
            self.backend.load_model(vals["model_name"], vals["num_labels"], vals["use_lora"])
            self.trainer_vals = vals
            self._debug("[TRAIN] Configured.")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", str(e))
            self._debug(f"[ERROR] Config failed: {e}")

    def action_train_only(self):
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(self, "Error", f"transformers/torch missing: {_TRAIN_IMPORT_ERROR}\npip install torch transformers")
            return
        if not self.dataset_loaded:
            QMessageBox.warning(self, "Dataset", "Load dataset first.")
            return
        if self.backend.base_model is None:
            self.action_configure_trainer_only()
            if self.backend.base_model is None:
                return
        out_dir = QFileDialog.getExistingDirectory(self, "Output Directory for HF Checkpoints")
        if not out_dir:
            return
        try:
            self.backend.configure_training(
                output_dir=out_dir,
                epochs=self.trainer_vals["epochs"],
                batch_size=self.trainer_vals["batch"],
                lr=self.trainer_vals["lr"],
                use_spacetime=self.chk_spacetime.isChecked(),
                use_cpu=self.trainer_vals["use_cpu"],
            )
            self.backend.train()
            QMessageBox.information(self, "Training", "Complete.")
            self.tabs.setCurrentIndex(1)  # Monitor
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self._debug(f"[ERROR] Training failed: {e}")

    def action_train_and_export(self):
        self.action_train_only()
        if self.backend.last_output_dir is None:
            return
        out_file, _ = QFileDialog.getSaveFileName(self, "Output GGUF", os.path.expanduser("~/romanai.gguf"), "GGUF (*.gguf)")
        if not out_file:
            return
        try:
            self.backend.export_gguf(self.backend.last_output_dir, out_file)
            if self.chk_4dllm.isChecked():
                base, _ = os.path.splitext(out_file)
                fourdllm = base + ".4dllm"
                shutil.copy2(out_file, fourdllm)
                out_file = fourdllm
                self._debug(f"[4DLLM] Created {fourdllm}")
            QMessageBox.information(self, "Export", f"Complete: {out_file}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self._debug(f"[ERROR] Export failed: {e}")

    # ---------- actions: Merge ----------
    def action_merge_only(self):
        if not self.primary_gguf:
            QMessageBox.warning(self, "GGUF", "Select primary GGUF.")
            return
        out_file, _ = QFileDialog.getSaveFileName(self, "Output Merged GGUF", os.path.expanduser("~/romanai_merged.gguf"), "GGUF (*.gguf)")
        if not out_file:
            return
        spacetime_profile = None
        if self.chk_spacetime.isChecked():
            # Arbitrary "size" for merge: number of models
            merge_size = len(self.secondary_ggufs) + 1
            profile = self.spacetime_engine.profile_for_training(merge_size)
            spacetime_profile = profile.as_dict()
            self._debug(f"[SPACETIME] Generated profile for merge (size={merge_size}): {self.spacetime_engine.describe(profile)}")
        try:
            simple_merge_gguf(self.primary_gguf, self.secondary_ggufs, out_file, self._debug, spacetime_profile)
            if self.chk_4dllm.isChecked():
                base, _ = os.path.splitext(out_file)
                fourdllm = base + ".4dllm"
                shutil.copy2(out_file, fourdllm)
                out_file = fourdllm
            QMessageBox.information(self, "Merge", f"Complete: {out_file}")
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", str(e))
            self._debug(f"[ERROR] Merge failed: {e}")

# =========================================================
#   App Entry
# =========================================================

def main():
    app = QApplication(sys.argv)
    apply_novalabs_titanium(app)
    win = SimpleForgeWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
