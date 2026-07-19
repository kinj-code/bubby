"""Settings dialog for Bubby — manage LLM path, TTS, autonomy, and memory stats."""

import logging, os, json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QFileDialog, QGroupBox, QMessageBox,
    QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class SettingsWindow(QDialog):
    """Standalone settings dialog for Bubby configuration."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bubby Settings")
        self.setMinimumSize(420, 320)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowCloseButtonHint
        )
        self._config = load_config()
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── LLM Model Path ───────────────────────────────────────
        llm_group = QGroupBox("LLM Model")
        llm_layout = QVBoxLayout(llm_group)

        path_row = QHBoxLayout()
        self._llm_path_edit = QLineEdit()
        self._llm_path_edit.setPlaceholderText("models/llm/model.gguf")
        path_row.addWidget(self._llm_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_model)
        path_row.addWidget(browse_btn)
        llm_layout.addLayout(path_row)

        self._llm_status = QLabel("")
        self._llm_status.setStyleSheet("color: #aaa; font-size: 11px;")
        llm_layout.addWidget(self._llm_status)

        layout.addWidget(llm_group)

        # ── Toggles ──────────────────────────────────────────────
        toggles_group = QGroupBox("Features")
        toggles_layout = QVBoxLayout(toggles_group)

        self._tts_check = QCheckBox("Enable Text-to-Speech (TTS)")
        toggles_layout.addWidget(self._tts_check)

        self._autonomy_check = QCheckBox("Enable Autonomy Loop (wander, observe)")
        toggles_layout.addWidget(self._autonomy_check)

        layout.addWidget(toggles_group)

        # ── Memory stats ─────────────────────────────────────────
        mem_group = QGroupBox("Memory")
        mem_layout = QVBoxLayout(mem_group)
        self._mem_label = QLabel("Records: —")
        mem_layout.addWidget(self._mem_label)
        layout.addWidget(mem_group)

        # ── Buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model File", "",
            "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self._llm_path_edit.setText(path)
            self._check_model()

    def _check_model(self) -> None:
        p = Path(self._llm_path_edit.text())
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            self._llm_status.setText(f"✓ Found ({size_mb:.0f} MB)")
            self._llm_status.setStyleSheet("color: #4caf50; font-size: 11px;")
        else:
            self._llm_status.setText("✗ File not found")
            self._llm_status.setStyleSheet("color: #f44336; font-size: 11px;")

    def _load_values(self) -> None:
        path = self._config.get("llm_path", os.environ.get("BUBBY_LLM_PATH", ""))
        self._llm_path_edit.setText(path)
        self._check_model()

        self._tts_check.setChecked(
            os.environ.get("BUBBY_USE_TTS", "0") == "1"
        )
        self._autonomy_check.setChecked(
            os.environ.get("BUBBY_USE_AUTONOMY", "1") == "1"
        )

        # Try to get memory stats from the running app via env
        records = os.environ.get("BUBBY_MEMORY_RECORDS", "")
        self._mem_label.setText(f"Records: {records}" if records else "Records: —")

    def _save(self) -> None:
        p = self._llm_path_edit.text().strip()
        if p and not Path(p).is_file():
            QMessageBox.warning(self, "Invalid Path", f"Model file not found:\n{p}")
            return

        cfg = dict(self._config)
        if p:
            cfg["llm_path"] = p
        else:
            cfg.pop("llm_path", None)
        save_config(cfg)

        # Also write to .env for compatibility
        env_path = CONFIG_PATH.parent / ".env"
        env_lines = []
        if env_path.exists():
            env_lines = env_path.read_text().splitlines()

        new_lines = []
        found_llm = found_tts = found_autonomy = False
        for line in env_lines:
            if line.startswith("BUBBY_LLM_PATH="):
                new_lines.append(f"BUBBY_LLM_PATH={p}")
                found_llm = True
            elif line.startswith("BUBBY_USE_TTS="):
                new_lines.append(f"BUBBY_USE_TTS={'1' if self._tts_check.isChecked() else '0'}")
                found_tts = True
            elif line.startswith("BUBBY_USE_AUTONOMY="):
                new_lines.append(f"BUBBY_USE_AUTONOMY={'1' if self._autonomy_check.isChecked() else '0'}")
                found_autonomy = True
            else:
                new_lines.append(line)
        if not found_llm and p:
            new_lines.append(f"BUBBY_LLM_PATH={p}")
        if not found_tts:
            new_lines.append(f"BUBBY_USE_TTS={'1' if self._tts_check.isChecked() else '0'}")
        if not found_autonomy:
            new_lines.append(f"BUBBY_USE_AUTONOMY={'1' if self._autonomy_check.isChecked() else '0'}")

        env_path.write_text("\n".join(new_lines) + "\n")
        logger.info("Settings saved to config.json and .env")
        self.accept()

    def set_memory_records(self, count: int) -> None:
        self._mem_label.setText(f"Records: {count}")