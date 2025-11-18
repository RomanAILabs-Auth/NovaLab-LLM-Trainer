#!/usr/bin/env python3
# =============================================================
# RomanAI – Simple GGUF Forge & 4DLLM Builder v1.0
# Project : RomanAILabs – NovaLabs 2.0 Simple Forge
# Purpose : Simple GGUF merger + dataset feeder + optional 4D/4DLLM
# Author  : Copyright (c) 2025 Daniel Harding – RomanAILabs
# Credits : Nova (GPT-5.1 Thinking, OpenAI), Grok (xAI)
#
# Notes:
#   - CPU-only trainer (no_cuda=True) for safety.
#   - "Merge" is a *logical* merge (primary GGUF copied, metadata logged).
#   - 4DLLM output is simply a renamed GGUF (your secret twist).
#   - Spacetime Engine is 4D-inspired and scales training hyperparams.
#   - FIXED: Dataset preview handles non-pandas cases gracefully.
#   - FIXED: GGUF export now auto-clones llama.cpp if root dir not found (dead simple).
#   - Run "pip install gitpython" for auto-clone feature.
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
    pd = None  # type: ignore

try:
    import numpy as np  # noqa: F401
except ImportError:
    np = None  # type: ignore

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
    torch = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    Trainer = None  # type: ignore
    TrainingArguments = None  # type: ignore
    TrainerCallback = object
    _TRAIN_IMPORT_ERROR = e
else:
    _TRAIN_IMPORT_ERROR = None

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
)
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

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
        self.log_fn("[MONITOR] Training started (CPU only).")

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
#   Model Backend – Simple CPU Trainer + GGUF Export
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
    def load_model(self, model_name: str, num_labels: int = 2):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(f"transformers/torch not available: {_TRAIN_IMPORT_ERROR}")
        self.debug(f"[TRAIN] Loading base HF model (CPU): {model_name}, labels={num_labels}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.feeder.set_tokenizer(self.tokenizer)
        self.debug("[TRAIN] Base model + tokenizer ready (CPU).")

    def configure_training(
        self,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        use_spacetime: bool = False,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(f"transformers/torch not available: {_TRAIN_IMPORT_ERROR}")
        if self.base_model is None or self.tokenizer is None:
            raise RuntimeError("Base model/tokenizer not loaded.")
        if self.feeder.dataset is None:
            raise RuntimeError("Dataset not loaded.")

        self.last_output_dir = output_dir

        scaled_epochs = epochs
        scaled_batch = batch_size
        scaled_lr = lr

        if use_spacetime:
            profile = self.spacetime.profile_for_training(self.dataset_size or 1)
            self.last_profile = profile
            self.debug(f"[SPACETIME] {self.spacetime.describe(profile)}")
            scaled_epochs = max(1, int(round(epochs * profile.depth_loops)))
            scaled_batch = max(1, int(round(batch_size * profile.batch_scale)))
            scaled_lr = float(lr * profile.lr_scale)
        else:
            self.last_profile = None
            self.debug("[SPACETIME] Disabled for this run.")

        self.debug(
            f"[TRAIN] Config: epochs={scaled_epochs}, batch={scaled_batch}, lr={scaled_lr:.2e}, "
            f"output_dir={output_dir}"
        )

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=scaled_epochs,
            per_device_train_batch_size=scaled_batch,
            learning_rate=scaled_lr,
            logging_steps=50,
            save_steps=1000,
            fp16=False,
            no_cuda=True,   # CPU only
            report_to=[],
        )

    def train(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(f"transformers/torch not available: {_TRAIN_IMPORT_ERROR}")
        if self.training_args is None:
            raise RuntimeError("TrainingArguments not configured.")
        if self.base_model is None or self.tokenizer is None:
            raise RuntimeError("Base model/tokenizer not loaded.")
        if self.feeder.dataset is None:
            raise RuntimeError("Dataset not loaded.")

        train_ds = self.feeder.get_torch_dataset()
        callback = UIMonitorCallback(self.monitor)

        self.trainer = Trainer(
            model=self.base_model,
            args=self.training_args,
            train_dataset=train_ds,
            callbacks=[callback],
        )

        self.debug("[TRAIN] Starting trainer.train() (CPU only)…")
        self.monitor("[MONITOR] Launching training on CPU…")
        self.trainer.train()
        self.monitor("[MONITOR] Training completed on CPU.")
        self.debug("[TRAIN] Training finished successfully.")

    def export_gguf(self, hf_dir: str, out_path: str):
        if git is None:
            raise RuntimeError("gitpython not installed. pip install gitpython for auto-clone.")

        llama_cpp_dir = "llama.cpp"
        if not os.path.exists(llama_cpp_dir):
            self.debug(f"[GGUF] llama.cpp not found - auto-cloning...")
            git.Repo.clone_from("https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir)
            self.debug("[GGUF] Cloned llama.cpp.")

        script_path = os.path.join(llama_cpp_dir, "convert.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError("convert.py not found in llama.cpp.")

        cmd = [sys.executable, script_path, hf_dir, "--outtype", "f16", "--outfile", out_path]
        self.debug(f"[GGUF] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        self.debug(f"[GGUF] Conversion complete: {out_path}")

# =========================================================
#   Feeder Config (System prompt + Persona)
# =========================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are RomanAI, a 4D spacetime-aware coding and reasoning assistant created "
    "by Daniel Harding (RomanAILabs). You think slowly where it matters and respond "
    "clearly, honestly, and in depth."
)

DEFAULT_PERSONA = (
    "Persona: Calm, structured, technical, no fluff. Explain tradeoffs, avoid hype, "
    "focus on practical gains and safety."
)

FEEDER_CONFIG_FILE = "romanai_feeder_config.json"


def load_feeder_config():
    if not os.path.exists(FEEDER_CONFIG_FILE):
        return {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "persona_prompt": DEFAULT_PERSONA,
        }
    with open(FEEDER_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feeder_config(system_prompt: str, persona_prompt: str):
    data = {
        "system_prompt": system_prompt,
        "persona_prompt": persona_prompt,
    }
    with open(FEEDER_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =========================================================
#   Simple GGUF Merger + 4DLLM twist
# =========================================================

def simple_merge_gguf(primary_path: str, secondary_paths: list, out_path: str, log_fn):
    """
    Very simple "merge":
      - Copy primary GGUF.
      - Record metadata indicating which secondaries were "merged" (just paths).
      - No real weight fusion; this is placeholder for advanced merges.
    """
    shutil.copy2(primary_path, out_path)
    log_fn(f"[MERGE] Copied primary GGUF to {out_path}")

    meta = {
        "merge_time": time.time(),
        "primary": primary_path,
        "secondaries": secondary_paths,
        "notes": "Simple copy-merge; no weight fusion.",
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log_fn(f"[MERGE] Metadata saved to {meta_path}")

# =========================================================
#   Dataset Wizard Dialog
# =========================================================

class DatasetWizardDialog(QDialog):
    def __init__(self, parent, path: str, fmt: str, feeder: DatasetFeeder):
        super().__init__(parent)
        self.setWindowTitle("Dataset Columns")
        self.path = path
        self.fmt = fmt
        self.feeder = feeder

        layout = QFormLayout(self)
        self.txt_text_col = QLineEdit("text")
        self.txt_label_col = QLineEdit("label")
        self.chk_label = QCheckBox("Has labels?")
        self.chk_label.toggled.connect(self._toggle_label)

        layout.addRow("Text column:", self.txt_text_col)
        layout.addRow(self.chk_label)
        layout.addRow("Label column:", self.txt_label_col)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

        self._toggle_label(self.chk_label.isChecked())

    def _toggle_label(self, checked: bool):
        self.txt_label_col.setEnabled(checked)

    def get_values(self) -> dict:
        if self.exec_() == QDialog.Accepted:
            label_col = self.txt_label_col.text().strip() if self.chk_label.isChecked() else None
            return {
                "text_col": self.txt_text_col.text().strip(),
                "label_col": label_col,
            }
        return {}

# =========================================================
#   Trainer Config Dialog
# =========================================================

class TrainerConfigDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Trainer Config")

        layout = QFormLayout(self)
        self.txt_model_name = QLineEdit("distilbert-base-uncased")
        self.txt_num_labels = QLineEdit("2")
        self.txt_epochs = QLineEdit("3")
        self.txt_batch = QLineEdit("8")
        self.txt_lr = QLineEdit("2e-5")

        layout.addRow("HF Base Model:", self.txt_model_name)
        layout.addRow("Num Labels:", self.txt_num_labels)
        layout.addRow("Epochs:", self.txt_epochs)
        layout.addRow("Batch Size:", self.txt_batch)
        layout.addRow("Learning Rate:", self.txt_lr)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_values(self) -> dict:
        if self.exec_() == QDialog.Accepted:
            return {
                "model_name": self.txt_model_name.text().strip(),
                "num_labels": int(self.txt_num_labels.text().strip() or "2"),
                "epochs": int(self.txt_epochs.text().strip() or "3"),
                "batch": int(self.txt_batch.text().strip() or "8"),
                "lr": float(self.txt_lr.text().strip() or "2e-5"),
            }
        return {}

# =========================================================
#   Main Window
# =========================================================

class SimpleForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RomanAI – Simple GGUF Forge & 4DLLM Builder v1.0")
        self.resize(1400, 800)

        self.backend = ModelBackend(SpacetimeEngine4D())
        self.backend.set_debugger(self._debug)
        self.backend.set_monitor(self._monitor)

        self.primary_gguf = None
        self.secondary_ggufs = []
        self.dataset_loaded = False
        self.spacetime_slider_value = 0.4
        self.spacetime_engine = SpacetimeEngine4D(global_dilation=0.4)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._load_feeder_text()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # LEFT: Simple Forge
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        simple_box = QWidget(self)
        simple_layout = QVBoxLayout(simple_box)
        simple_layout.setContentsMargins(6, 6, 6, 6)
        simple_layout.setSpacing(4)

        title = QLabel("Simple Forge", self)
        tfont = QFont()
        tfont.setPointSize(12)
        tfont.setBold(True)
        title.setFont(tfont)

        desc = QLabel(
            "1. Select Primary GGUF (base)\n"
            "2. Optional: Add Secondary GGUFs\n"
            "3. Merge Only or Train + Export\n"
            "4. Optional: 4D Spacetime / 4DLLM",
            self,
        )
        desc.setWordWrap(True)

        self.lbl_primary = QLabel("Primary GGUF: <none>", self)
        self.lbl_secondaries = QLabel("Secondary GGUFs: <none>", self)

        btn_primary = QPushButton("Select Primary GGUF", self)
        btn_primary.clicked.connect(self.action_select_primary)

        btn_secondaries = QPushButton("Add Secondary GGUFs", self)
        btn_secondaries.clicked.connect(self.action_select_secondaries)

        btn_clear_gguf = QPushButton("Clear GGUF Selections", self)
        btn_clear_gguf.clicked.connect(self.action_clear_ggufs)

        self.chk_spacetime = QCheckBox("Use 4D Spacetime Engine", self)
        self.chk_spacetime.setChecked(True)

        self.chk_4dllm = QCheckBox("Output as 4DLLM", self)
        self.chk_4dllm.setChecked(True)

        btn_merge = QPushButton("Merge Only (no train)", self)
        btn_merge.clicked.connect(self.action_merge_only)

        btn_train_only = QPushButton("Train Only (no GGUF export)", self)
        btn_train_only.clicked.connect(self.action_train_only)

        btn_train_export = QPushButton("Train + Export GGUF / 4DLLM", self)
        btn_train_export.clicked.connect(self.action_train_and_export)

        simple_layout.addWidget(title)
        simple_layout.addWidget(desc)
        simple_layout.addWidget(self.lbl_primary)
        simple_layout.addWidget(self.lbl_secondaries)
        simple_layout.addWidget(btn_primary)
        simple_layout.addWidget(btn_secondaries)
        simple_layout.addWidget(btn_clear_gguf)
        simple_layout.addSpacing(4)
        simple_layout.addWidget(self.chk_spacetime)
        simple_layout.addWidget(self.chk_4dllm)
        simple_layout.addSpacing(6)
        simple_layout.addWidget(btn_merge)
        simple_layout.addWidget(btn_train_only)
        simple_layout.addWidget(btn_train_export)
        simple_layout.addStretch(1)

        left_layout.addWidget(simple_box)

        # --- Logs ---
        logs_box = QWidget(self)
        logs_layout = QVBoxLayout(logs_box)
        logs_layout.setContentsMargins(4, 4, 4, 4)
        logs_layout.setSpacing(4)

        logs_label = QLabel("Debugger / Forge Log", self)
        lfont = QFont()
        lfont.setPointSize(10)
        lfont.setBold(True)
        logs_label.setFont(lfont)

        self.debugger = QPlainTextEdit(self)
        self.debugger.setReadOnly(True)
        self.debugger.setPlaceholderText("Logs will appear here…")

        logs_layout.addWidget(logs_label)
        logs_layout.addWidget(self.debugger)

        left_layout.addWidget(logs_box, 2)

        splitter.addWidget(left_widget)

        # RIGHT: Tabs
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.tabs = QTabWidget(self)

        # Dataset tab
        ds_widget = QWidget(self)
        ds_layout = QVBoxLayout(ds_widget)
        ds_layout.setContentsMargins(8, 8, 8, 8)
        ds_layout.setSpacing(6)

        ds_label = QLabel("Dataset Feeder", self)
        ds_font = QFont()
        ds_font.setPointSize(11)
        ds_font.setBold(True)
        ds_label.setFont(ds_font)

        self.btn_load_dataset = QPushButton("Load Dataset (CSV/JSON/TXT)…", self)
        self.btn_load_dataset.clicked.connect(self.action_load_dataset)

        self.dataset_info = QLabel("Dataset: <none>", self)

        self.preview_table = QTableWidget(self)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)

        ds_layout.addWidget(ds_label)
        ds_layout.addWidget(self.btn_load_dataset)
        ds_layout.addWidget(self.dataset_info)
        ds_layout.addWidget(self.preview_table)

        self.tabs.addTab(ds_widget, "Dataset")

        # Feeder tab – system prompt + persona
        feeder_widget = QWidget(self)
        feeder_layout = QVBoxLayout(feeder_widget)
        feeder_layout.setContentsMargins(8, 8, 8, 8)
        feeder_layout.setSpacing(6)

        feeder_label = QLabel("Feeder: System Prompt + Persona", self)
        ffont = QFont()
        ffont.setPointSize(11)
        ffont.setBold(True)
        feeder_label.setFont(ffont)

        self.txt_system_prompt = QPlainTextEdit(self)
        self.txt_persona = QPlainTextEdit(self)

        btn_save_feeder = QPushButton("Save Feeder Config", self)
        btn_save_feeder.clicked.connect(self.action_save_feeder)

        feeder_layout.addWidget(feeder_label)
        feeder_layout.addWidget(QLabel("System Prompt:", self))
        feeder_layout.addWidget(self.txt_system_prompt, 1)
        feeder_layout.addWidget(QLabel("Persona Prompt:", self))
        feeder_layout.addWidget(self.txt_persona, 1)
        feeder_layout.addWidget(btn_save_feeder)

        self.tabs.addTab(feeder_widget, "Feeder")

        # Spacetime HUD tab
        st_widget = QWidget(self)
        st_layout = QVBoxLayout(st_widget)
        st_layout.setContentsMargins(8, 8, 8, 8)
        st_layout.setSpacing(6)

        st_label = QLabel("Spacetime HUD (4D Engine)", self)
        st_font = QFont()
        st_font.setPointSize(11)
        st_font.setBold(True)
        st_label.setFont(st_font)

        self.st_slider = QSlider(Qt.Horizontal, self)
        self.st_slider.setMinimum(0)
        self.st_slider.setMaximum(100)
        self.st_slider.setValue(int(self.spacetime_slider_value * 100))
        self.st_slider.valueChanged.connect(self._on_spacetime_change)

        self.st_info = QLabel("Global dilation: 0.40", self)

        self.monitor_log = QPlainTextEdit(self)
        self.monitor_log.setReadOnly(True)
        self.monitor_log.setPlaceholderText("Training monitor (steps, loss, timings)…")

        st_layout.addWidget(st_label)
        st_layout.addWidget(self.st_slider)
        st_layout.addWidget(self.st_info)
        st_layout.addWidget(self.monitor_log)

        self.tabs.addTab(st_widget, "Spacetime HUD")

        # Info tab – keep some “good stuff”
        info_widget = QPlainTextEdit(self)
        info_widget.setReadOnly(True)
        info_widget.setPlainText(
            "RomanAI Simple Forge – NovaLabs 2.0\n\n"
            "This is the simplified front-end:\n"
            "  • GGUF merger (primary + optional secondaries)\n"
            "  • 4D Spacetime Engine (training hyperparams)\n"
            "  • Optional 4DLLM output (renamed GGUF)\n"
            "  • Dataset Feeder + System/Persona prompts\n\n"
            "Advanced ideas (for later versions):\n"
            "  • QLoRA on GGUF\n"
            "  • Router RLHF between multiple GGUF models\n"
            "  • Dual Master/Scriptor fusion\n"
            "  • Live API wiring to ChatGPT / remote coders\n\n"
            "For now, everything is kept simple and CPU-safe."
        )

        self.tabs.addTab(info_widget, "Info")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)
        splitter.setSizes([520, 880])

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
                    item.setFlags(Qt.ItemIsEnabled)  # Read-only
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
            item = QTableWidgetItem("No preview available for this format.")
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
        fmt = "csv"
        if ext == ".json":
            fmt = "json"
        elif ext == ".txt":
            fmt = "txt"

        wizard = DatasetWizardDialog(self, path, fmt, self.backend.feeder)
        vals = wizard.get_values()
        if not vals:
            return
        text_col = vals["text_col"]
        label_col = vals["label_col"]

        try:
            size = self.backend.load_dataset(path, fmt, text_col, label_col)
        except Exception as e:
            QMessageBox.critical(self, "Dataset Error", str(e))
            self._debug(f"[ERROR] Dataset load failed: {e}")
            return

        self.dataset_loaded = True
        self.dataset_format = fmt
        self.dataset_info.setText(f"Dataset: {os.path.basename(path)} ({size} items)")
        self._debug(f"[DATASET] Loaded {path} ({size} items) text={text_col}, label={label_col}")
        self._refresh_dataset_preview()
        self.tabs.setCurrentIndex(0)

    # ---------- actions: Feeder ----------
    def action_save_feeder(self):
        sp = self.txt_system_prompt.toPlainText().strip()
        pp = self.txt_persona.toPlainText().strip()
        if not sp:
            sp = DEFAULT_SYSTEM_PROMPT
        if not pp:
            pp = DEFAULT_PERSONA
        save_feeder_config(sp, pp)
        self._debug("[FEEDER] Saved system/persona prompts to romanai_feeder_config.json")
        QMessageBox.information(self, "Feeder", "Feeder config saved.")

    # ---------- actions: Trainer ----------
    def action_configure_trainer_only(self):
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(
                self,
                "Dependencies missing",
                f"transformers/torch not available:\n{_TRAIN_IMPORT_ERROR}\n\n"
                "Install with:\n  pip install torch transformers",
            )
            return
        if not self.dataset_loaded:
            QMessageBox.warning(self, "Dataset", "Please load a dataset first.")
            return

        dlg = TrainerConfigDialog(self)
        vals = dlg.get_values()
        if not vals:
            return

        try:
            self.backend.load_model(vals["model_name"], vals["num_labels"])
            self.trainer_vals = vals
        except Exception as e:
            QMessageBox.critical(self, "Trainer Config Error", str(e))
            self._debug(f"[ERROR] Config failed: {e}")
            return

        self._debug("[TRAIN] Trainer configured (CPU only).")
        QMessageBox.information(
            self,
            "Trainer",
            "Trainer configured. Use 'Train Only' or 'Train + Export'.",
        )

    def action_train_only(self):
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(
                self,
                "Dependencies missing",
                f"transformers/torch not available:\n{_TRAIN_IMPORT_ERROR}\n\n"
                "Install with:\n  pip install torch transformers",
            )
            return
        if self.backend.base_model is None:
            self.action_configure_trainer_only()
            if self.backend.base_model is None:
                return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Directory for HF Checkpoints",
            "",
        )
        if not out_dir:
            return

        try:
            self.backend.configure_training(
                output_dir=out_dir,
                epochs=self.trainer_vals["epochs"],
                batch_size=self.trainer_vals["batch"],
                lr=self.trainer_vals["lr"],
                use_spacetime=self.chk_spacetime.isChecked(),
            )
            self.backend.train()
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self._debug(f"[ERROR] Training failed: {e}")
            return
        QMessageBox.information(self, "Training", "Training complete on CPU.")
        self.tabs.setCurrentIndex(2)

    def action_train_and_export(self):
        # 1) Train (using same path as Train Only)
        self.action_train_only()
        if self.backend.last_output_dir is None:
            return

        # 2) Choose GGUF output
        default_out = os.path.join(os.path.expanduser("~"), "romanai_simple.gguf")
        out_file, _ = QFileDialog.getSaveFileName(
            self,
            "Output GGUF File",
            default_out,
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if not out_file:
            return

        try:
            self.backend.export_gguf(self.backend.last_output_dir, out_file)
        except Exception as e:
            QMessageBox.critical(self, "GGUF Error", str(e))
            self._debug(f"[ERROR] GGUF conversion failed: {e}")
            return

        final_path = out_file
        if self.chk_4dllm.isChecked():
            base, _ = os.path.splitext(out_file)
            fourdllm = base + ".4dllm"
            shutil.copy2(out_file, fourdllm)
            final_path = fourdllm
            self._debug(f"[4DLLM] Copied GGUF → {fourdllm}")
            QMessageBox.information(
                self,
                "4DLLM",
                f"GGUF exported and mirrored as 4DLLM:\n{fourdllm}",
            )
        else:
            QMessageBox.information(
                self,
                "GGUF",
                f"GGUF export complete:\n{out_file}",
            )

        self._debug(f"[EXPORT] Final model file: {final_path}")
        self.tabs.setCurrentIndex(2)

    # ---------- actions: Merge ----------
    def action_merge_only(self):
        if not self.primary_gguf:
            QMessageBox.warning(self, "GGUF", "Select a primary GGUF first.")
            return

        default_name = os.path.join(
            os.path.expanduser("~"),
            "romanai_merged.gguf",
        )
        out_file, _ = QFileDialog.getSaveFileName(
            self,
            "Output Merged GGUF / 4DLLM",
            default_name,
            "GGUF / 4DLLM (*.gguf *.4dllm);;All Files (*)",
        )
        if not out_file:
            return

        # Merge (copy + metadata)
        try:
            simple_merge_gguf(self.primary_gguf, self.secondary_ggufs, out_file, self._debug)
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", str(e))
            self._debug(f"[ERROR] Merge failed: {e}")
            return

        final_path = out_file
        if self.chk_4dllm.isChecked():
            base, _ = os.path.splitext(out_file)
            fourdllm = base + ".4dllm"
            shutil.copy2(out_file, fourdllm)
            final_path = fourdllm
            self._debug(f"[4DLLM] Renamed/copy to {fourdllm}")

        QMessageBox.information(
            self,
            "Merge",
            f"Merge complete.\nFinal file: {final_path}",
        )

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
