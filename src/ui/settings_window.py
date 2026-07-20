"""
Comprehensive Settings GUI for Bubby — AI, API, Voice, Character management.

SPRINT 2: Self-Contained Settings GUI & LLM Management

Features:
- AI Settings: Temperature slider, context length, persona selection
- API Management: OpenAI/Anthropic API keys with hybrid mode support
- Voice Settings: Mute toggle, volume slider, voice model download/import
- Character Settings: Character model/skin selection
- Performance: System spec analyzer (RAM/VRAM), Efficient Mode
- LLM Hosting: llama-cpp-python managed internally
"""

import logging
import os
import json
import platform
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QFileDialog, QGroupBox, QMessageBox,
    QTabWidget, QSlider, QSpinBox, QComboBox, QGridLayout, QProgressBar,
    QTextEdit, QScrollArea, QRadioButton, QButtonGroup,
    QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QFont, QIcon

from src.persona.config import PersonaType, PersonaConfig

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


# ── System Spec Analyzer ──────────────────────────────────────────

@dataclass
class SystemSpecs:
    """System hardware specifications."""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    vram_gb: float = 0.0
    cpu_cores: int = 0
    has_nvidia_gpu: bool = False
    has_amd_gpu: bool = False

    def recommended_model_size(self) -> str:
        """Suggest model size based on available RAM/VRAM."""
        usable = self.vram_gb if self.vram_gb > 0 else self.available_ram_gb
        if usable >= 16:
            return "8B"  # Can run 8B model comfortably
        elif usable >= 8:
            return "7B"  # Can run 7B model
        elif usable >= 4:
            return "3B"  # Can run 3B model
        else:
            return "1B"  # Small model

    def efficient_mode_recommended(self) -> bool:
        """Whether efficient mode (3B) is recommended."""
        usable = self.vram_gb if self.vram_gb > 0 else self.available_ram_gb
        return usable < 8


def detect_system_specs() -> SystemSpecs:
    """Detect system hardware specs for model recommendations."""
    specs = SystemSpecs()

    # RAM detection
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        specs.total_ram_gb = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith("MemAvailable:"):
                        specs.available_ram_gb = int(line.split()[1]) / 1024 / 1024
    except Exception:
        specs.total_ram_gb = 8.0  # Fallback
        specs.available_ram_gb = 4.0

    # CPU cores
    try:
        specs.cpu_cores = os.cpu_count() or 4
    except Exception:
        specs.cpu_cores = 4

    # NVIDIA GPU VRAM via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    specs.vram_gb = float(line.strip()) / 1024
                    specs.has_nvidia_gpu = True
                    break
    except Exception:
        pass

    # AMD GPU check
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            specs.has_amd_gpu = True
    except Exception:
        pass

    return specs


# ── Voice Model Download Worker ───────────────────────────────────

class VoiceModelDownloader(QThread):
    """Background thread for downloading voice models."""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, model_name: str, target_dir: Path) -> None:
        super().__init__()
        self._model_name = model_name
        self._target_dir = target_dir

    def run(self) -> None:
        """Download a voice model in the background."""
        try:
            self.status.emit(f"Downloading {self._model_name}...")
            import urllib.request
            import shutil

            self._target_dir.mkdir(parents=True, exist_ok=True)

            # Map common voice model names to URLs
            voice_urls = {
                "amy": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
                "norman": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/low/en_US-norman-low.onnx",
                "ljspeech": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/low/en_US-ljspeech-low.onnx",
            }

            url = voice_urls.get(self._model_name)
            if not url:
                self.finished.emit(False, f"Unknown model: {self._model_name}")
                return

            file_path = self._target_dir / f"{self._model_name}.onnx"

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    pct = int(block_num * block_size * 100 / total_size)
                    self.progress.emit(min(pct, 100))

            urllib.request.urlretrieve(url, file_path, reporthook)
            self.finished.emit(True, f"Downloaded {self._model_name}")
        except Exception as e:
            self.finished.emit(False, str(e))


# ── Comprehensive Settings Window ─────────────────────────────────

class SettingsWindow(QMainWindow):
    """
    Comprehensive settings control panel for Bubby.

    Tabs:
    1. AI Settings — Temperature, context length, persona
    2. API Management — OpenAI/Anthropic keys
    3. Voice Settings — TTS, volume, voice models
    4. Character — Model/skin selection
    5. Performance — System specs, efficient mode
    6. About — Version, credits
    """

    # Signal emitted when settings change (for live updates)
    settings_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bubby Settings")
        self.setMinimumSize(560, 480)
        self.resize(640, 540)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        self._config = load_config()
        self._system_specs = detect_system_specs()
        self._voice_downloader: Optional[VoiceModelDownloader] = None

        self._build_ui()
        self._load_values()

        logger.info("SettingsWindow opened")

    def _build_ui(self) -> None:
        """Build the settings UI with tabs."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # ── Tab 1: AI Settings ──
        self._build_ai_tab()
        # ── Tab 2: API Management ──
        self._build_api_tab()
        # ── Tab 3: Voice Settings ──
        self._build_voice_tab()
        # ── Tab 4: Character ──
        self._build_character_tab()
        # ── Tab 5: Performance ──
        self._build_performance_tab()
        # ── Tab 6: About ──
        self._build_about_tab()

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        save_btn = QPushButton("Save & Close")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #4caf50; color: white; padding: 8px 24px;
                border-radius: 6px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #45a049; }
        """)
        save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #555; color: white; padding: 8px 24px;
                border-radius: 6px; font-size: 13px;
            }
            QPushButton:hover { background: #666; }
        """)
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _build_ai_tab(self) -> None:
        """AI Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── Temperature ──
        temp_group = QGroupBox("Temperature")
        temp_layout = QVBoxLayout(temp_group)

        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Creativity:"))
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(0, 100)
        self._temp_slider.setValue(70)
        self._temp_value = QLabel("0.70")
        self._temp_value.setMinimumWidth(40)
        self._temp_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_value.setText(f"{v/100:.2f}")
        )
        temp_row.addWidget(self._temp_slider)
        temp_row.addWidget(self._temp_value)
        temp_layout.addLayout(temp_row)

        temp_hint = QLabel("Lower = more deterministic, Higher = more creative")
        temp_hint.setStyleSheet("color: #999; font-size: 11px;")
        temp_layout.addWidget(temp_hint)
        layout.addWidget(temp_group)

        # ── Context Length ──
        ctx_group = QGroupBox("Context Length")
        ctx_layout = QVBoxLayout(ctx_group)

        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Max tokens:"))
        self._context_spin = QSpinBox()
        self._context_spin.setRange(512, 8192)
        self._context_spin.setSingleStep(512)
        self._context_spin.setValue(2048)
        self._context_spin.setSuffix(" tokens")
        ctx_row.addWidget(self._context_spin)
        ctx_row.addStretch()
        ctx_layout.addLayout(ctx_row)

        ctx_hint = QLabel("Higher = remembers more, uses more RAM")
        ctx_hint.setStyleSheet("color: #999; font-size: 11px;")
        ctx_layout.addWidget(ctx_hint)
        layout.addWidget(ctx_group)

        # ── Persona ──
        persona_group = QGroupBox("Persona")
        persona_layout = QVBoxLayout(persona_group)

        persona_row = QHBoxLayout()
        persona_row.addWidget(QLabel("Character:"))
        self._persona_combo = QComboBox()
        self._persona_combo.addItems([
            PersonaType.WITTY_COMPANION.value,
            PersonaType.HELPFUL_COPILOT.value,
            PersonaType.MINIMALIST.value,
        ])
        self._persona_combo.setCurrentText(PersonaType.WITTY_COMPANION.value)
        self._persona_combo.currentTextChanged.connect(self._on_persona_changed)
        persona_row.addWidget(self._persona_combo)
        persona_row.addStretch()
        persona_layout.addLayout(persona_row)

        self._persona_desc = QLabel(
            "Warm, playful, slightly witty — the default Bubby experience"
        )
        self._persona_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        self._persona_desc.setWordWrap(True)
        persona_layout.addWidget(self._persona_desc)
        layout.addWidget(persona_group)

        # ── Features ──
        feat_group = QGroupBox("Features")
        feat_layout = QVBoxLayout(feat_group)
        self._tts_check = QCheckBox("Enable Text-to-Speech (TTS)")
        self._tts_check.setToolTip("Requires a Piper TTS voice model")
        feat_layout.addWidget(self._tts_check)
        self._autonomy_check = QCheckBox("Enable Autonomy (wander, observe)")
        feat_layout.addWidget(self._autonomy_check)
        self._vision_check = QCheckBox("Enable Screen Awareness (Vision)")
        feat_layout.addWidget(self._vision_check)
        layout.addWidget(feat_group)

        layout.addStretch()
        self._tabs.addTab(tab, "🤖 AI Settings")

    def _build_api_tab(self) -> None:
        """API Management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── OpenAI ──
        openai_group = QGroupBox("OpenAI API")
        openai_layout = QVBoxLayout(openai_group)

        openai_row = QHBoxLayout()
        openai_row.addWidget(QLabel("API Key:"))
        self._openai_key = QLineEdit()
        self._openai_key.setPlaceholderText("sk-...")
        self._openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        openai_row.addWidget(self._openai_key)
        openai_layout.addLayout(openai_row)

        show_btn = QPushButton("Show")
        show_btn.setFixedWidth(60)
        show_btn.clicked.connect(
            lambda: self._openai_key.setEchoMode(
                QLineEdit.EchoMode.Normal if self._openai_key.echoMode() == QLineEdit.EchoMode.Password
                else QLineEdit.EchoMode.Password
            )
        )
        openai_row.addWidget(show_btn)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._openai_model = QComboBox()
        self._openai_model.addItems(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
        self._openai_model.setCurrentText("gpt-4o-mini")
        model_row.addWidget(self._openai_model)
        model_row.addStretch()
        openai_layout.addLayout(model_row)

        openai_hint = QLabel("Enable hybrid mode: uses local LLM by default, falls back to API")
        openai_hint.setStyleSheet("color: #999; font-size: 11px;")
        openai_hint.setWordWrap(True)
        openai_layout.addWidget(openai_hint)
        layout.addWidget(openai_group)

        # ── Anthropic ──
        anthropic_group = QGroupBox("Anthropic API")
        anthropic_layout = QVBoxLayout(anthropic_group)

        anthropic_row = QHBoxLayout()
        anthropic_row.addWidget(QLabel("API Key:"))
        self._anthropic_key = QLineEdit()
        self._anthropic_key.setPlaceholderText("sk-ant-...")
        self._anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        anthropic_row.addWidget(self._anthropic_key)

        show_btn2 = QPushButton("Show")
        show_btn2.setFixedWidth(60)
        show_btn2.clicked.connect(
            lambda: self._anthropic_key.setEchoMode(
                QLineEdit.EchoMode.Normal if self._anthropic_key.echoMode() == QLineEdit.EchoMode.Password
                else QLineEdit.EchoMode.Password
            )
        )
        anthropic_row.addWidget(show_btn2)
        anthropic_layout.addLayout(anthropic_row)

        model_row2 = QHBoxLayout()
        model_row2.addWidget(QLabel("Model:"))
        self._anthropic_model = QComboBox()
        self._anthropic_model.addItems(["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022"])
        self._anthropic_model.setCurrentText("claude-haiku-3-5-20241022")
        model_row2.addWidget(self._anthropic_model)
        model_row2.addStretch()
        anthropic_layout.addLayout(model_row2)

        # Hybrid mode toggle
        self._hybrid_mode = QCheckBox("Hybrid Mode (local LLM + API fallback)")
        self._hybrid_mode.setToolTip("Use local LLM by default, fall back to API when local is unavailable")
        anthropic_layout.addWidget(self._hybrid_mode)
        layout.addWidget(anthropic_group)

        # ── Local LLM Path ──
        llm_group = QGroupBox("Local LLM")
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

        # Download button
        dl_row = QHBoxLayout()
        dl_row.addWidget(QLabel("Download model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "Llama-3.2-3B-Instruct (recommended)",
            "Llama-3.2-1B-Instruct (fast)",
            "Mistral-7B-Instruct (powerful)",
            "Phi-3-mini-3.8B",
        ])
        dl_row.addWidget(self._model_combo)
        dl_btn = QPushButton("Download")
        dl_btn.clicked.connect(self._download_model)
        dl_row.addWidget(dl_btn)
        llm_layout.addLayout(dl_row)
        self._dl_progress = QProgressBar()
        self._dl_progress.setVisible(False)
        llm_layout.addWidget(self._dl_progress)

        layout.addWidget(llm_group)
        layout.addStretch()
        self._tabs.addTab(tab, "🔑 API Keys")

    def _build_voice_tab(self) -> None:
        """Voice Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── TTS Toggle ──
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QVBoxLayout(tts_group)

        self._mute_check = QCheckBox("Mute (disable all speech)")
        tts_layout.addWidget(self._mute_check)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(80)
        vol_row.addWidget(self._volume_slider)
        self._vol_value = QLabel("80%")
        self._vol_value.setMinimumWidth(40)
        self._volume_slider.valueChanged.connect(
            lambda v: self._vol_value.setText(f"{v}%")
        )
        vol_row.addWidget(self._vol_value)
        tts_layout.addLayout(vol_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(50, 200)
        self._speed_slider.setValue(100)
        speed_row.addWidget(self._speed_slider)
        self._speed_value = QLabel("1.0x")
        self._speed_value.setMinimumWidth(40)
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_value.setText(f"{v/100:.1f}x")
        )
        speed_row.addWidget(self._speed_value)
        tts_layout.addLayout(speed_row)
        layout.addWidget(tts_group)

        # ── Voice Models ──
        voice_group = QGroupBox("Voice Models")
        voice_layout = QVBoxLayout(voice_group)

        # Voice list
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Active voice:"))
        self._voice_combo = QComboBox()
        self._voice_combo.addItems(["amy", "norman", "ljspeech"])
        self._voice_combo.setCurrentText("amy")
        voice_row.addWidget(self._voice_combo)

        test_btn = QPushButton("Test")
        test_btn.clicked.connect(self._test_voice)
        voice_row.addWidget(test_btn)
        voice_layout.addLayout(voice_row)

        # Import voice model from file
        import_row = QHBoxLayout()
        import_btn = QPushButton("Import Voice Model...")
        import_btn.clicked.connect(self._import_voice_model)
        import_row.addWidget(import_btn)

        download_voice_btn = QPushButton("Download Voice")
        download_voice_btn.clicked.connect(self._download_voice_model)
        import_row.addWidget(download_voice_btn)
        import_row.addStretch()
        voice_layout.addLayout(import_row)

        self._voice_progress = QProgressBar()
        self._voice_progress.setVisible(False)
        voice_layout.addWidget(self._voice_progress)

        self._voice_status = QLabel("")
        self._voice_status.setStyleSheet("color: #aaa; font-size: 11px;")
        voice_layout.addWidget(self._voice_status)
        layout.addWidget(voice_group)

        layout.addStretch()
        self._tabs.addTab(tab, "🎤 Voice")

    def _build_character_tab(self) -> None:
        """Character Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── Character Appearance ──
        char_group = QGroupBox("Character Appearance")
        char_layout = QVBoxLayout(char_group)

        skin_row = QHBoxLayout()
        skin_row.addWidget(QLabel("Skin/Theme:"))
        self._skin_combo = QComboBox()
        self._skin_combo.addItems(["Default Slime 🫧", "Night Mode 🌙", "Pastel 🌸", "Retro 🕹️"])
        skin_row.addWidget(self._skin_combo)
        skin_row.addStretch()
        char_layout.addLayout(skin_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Window Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(100, 400)
        self._size_slider.setValue(400)
        size_row.addWidget(self._size_slider)
        self._size_value = QLabel("400px")
        self._size_value.setMinimumWidth(50)
        self._size_slider.valueChanged.connect(
            lambda v: self._size_value.setText(f"{v}px")
        )
        size_row.addWidget(self._size_value)
        char_layout.addLayout(size_row)

        # Sprite pack
        sprite_row = QHBoxLayout()
        sprite_row.addWidget(QLabel("Animation Style:"))
        self._sprite_combo = QComboBox()
        self._sprite_combo.addItems(["Emoji (built-in)", "Sprite PNG (if available)", "GIF (if available)"])
        sprite_row.addWidget(self._sprite_combo)
        sprite_row.addStretch()
        char_layout.addLayout(sprite_row)

        layout.addWidget(char_group)

        # ── Behavior ──
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_group)

        self._bob_check = QCheckBox("Gentle floating animation (bob)")
        self._bob_check.setChecked(True)
        behavior_layout.addWidget(self._bob_check)

        self._wander_check = QCheckBox("Idle wandering around screen")
        self._wander_check.setChecked(True)
        behavior_layout.addWidget(self._wander_check)

        self._popup_check = QCheckBox("Show chat popup on hover")
        self._popup_check.setChecked(True)
        behavior_layout.addWidget(self._popup_check)

        layout.addWidget(behavior_group)
        layout.addStretch()
        self._tabs.addTab(tab, "🎨 Character")

    def _build_performance_tab(self) -> None:
        """Performance tab with system spec analyzer."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── System Specs ──
        sys_group = QGroupBox("System Specifications")
        sys_layout = QVBoxLayout(sys_group)

        specs = self._system_specs
        sys_info = QTextEdit()
        sys_info.setReadOnly(True)
        sys_info.setMaximumHeight(160)
        sys_info.setStyleSheet("""
            QTextEdit {
                background: #1a1a2e;
                color: #ccc;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
            }
        """)

        info_text = (
            f"🧠 RAM:     {specs.total_ram_gb:.1f} GB total, {specs.available_ram_gb:.1f} GB available\n"
            f"🎮 VRAM:    {specs.vram_gb:.1f} GB {'(NVIDIA GPU)' if specs.has_nvidia_gpu else '(not detected)'}\n"
            f"⚙️  CPU:     {specs.cpu_cores} cores\n"
            f"📦 Platform: {platform.platform()}\n\n"
            f"💡 Recommended model: {specs.recommended_model_size()}\n"
        )
        if specs.efficient_mode_recommended():
            info_text += f"   ⚠️  Efficient mode recommended (3B model)\n"
        else:
            info_text += f"   ✓ Full mode available (7B+ model)\n"

        sys_info.setText(info_text)
        sys_layout.addWidget(sys_info)

        refresh_btn = QPushButton("🔄 Refresh System Specs")
        refresh_btn.clicked.connect(self._refresh_specs)
        sys_layout.addWidget(refresh_btn)
        layout.addWidget(sys_group)

        # ── Efficient Mode ──
        eff_group = QGroupBox("Efficient Mode")
        eff_layout = QVBoxLayout(eff_group)

        self._efficient_mode = QCheckBox("Efficient Mode (lower RAM usage)")
        self._efficient_mode.setToolTip(
            "Reduces model size to 3B parameters, lowers context length, "
            "disables vision pipeline to save memory"
        )
        eff_layout.addWidget(self._efficient_mode)

        eff_desc = QLabel(
            f"Recommended: {specs.efficient_mode_recommended()}"
        )
        eff_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        eff_layout.addWidget(eff_desc)

        # Auto-suggest
        if specs.efficient_mode_recommended():
            self._efficient_mode.setChecked(True)
            eff_desc.setText(
                "✓ ENABLED — Your system has limited RAM/VRAM. "
                "Efficient mode will prevent crashes."
            )

        layout.addWidget(eff_group)

        # ── Memory Stats ──
        mem_group = QGroupBox("Memory Usage")
        mem_layout = QVBoxLayout(mem_group)

        self._mem_label = QLabel("Long-term memory records: —")
        mem_layout.addWidget(self._mem_label)

        clear_mem_btn = QPushButton("Clear Memory Cache")
        clear_mem_btn.clicked.connect(self._clear_memory)
        mem_layout.addWidget(clear_mem_btn)
        layout.addWidget(mem_group)

        layout.addStretch()
        self._tabs.addTab(tab, "⚡ Performance")

    def _build_about_tab(self) -> None:
        """About tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("🫧 Bubby")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #8ab4f8;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version = QLabel("v2.0 — \"Genshin Pet\" Edition")
        version.setStyleSheet("font-size: 16px; color: #aaa;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        layout.addSpacing(20)

        desc = QLabel(
            "A friendly desktop companion that lives on your screen.\n"
            "Built with PySide6 — powered by local LLMs.\n\n"
            "Features:\n"
            "• Always-on-top transparent overlay\n"
            "• Hover-to-chat popup\n"
            "• Drag-and-drop file support\n"
            "• Screen awareness (vision)\n"
            "• Text-to-speech\n"
            "• 33+ emotions and animations"
        )
        desc.setStyleSheet("color: #888; font-size: 13px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        layout.addSpacing(20)

        credits = QLabel("Created by @kinj-code © 2026")
        credits.setStyleSheet("color: #666; font-size: 11px;")
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(credits)

        layout.addStretch()
        self._tabs.addTab(tab, "ℹ️ About")

    # ── Slots ──

    def _on_persona_changed(self, persona: str) -> None:
        """Update persona description when selection changes."""
        descs = {
            PersonaType.WITTY_COMPANION.value: "Warm, playful, slightly witty — the default Bubby experience",
            PersonaType.HELPFUL_COPILOT.value: "Professional, concise, focused on productivity",
            PersonaType.MINIMALIST.value: "Minimal responses, no-nonsense, efficient",
        }
        self._persona_desc.setText(descs.get(persona, ""))

    def _browse_model(self) -> None:
        """Browse for a GGUF model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model File", "",
            "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self._llm_path_edit.setText(path)
            self._check_model()

    def _check_model(self) -> None:
        """Check if the model file exists and show status."""
        p = Path(self._llm_path_edit.text())
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            self._llm_status.setText(f"✓ Found ({size_mb:.0f} MB)")
            self._llm_status.setStyleSheet("color: #4caf50; font-size: 11px;")
        else:
            self._llm_status.setText("✗ File not found")
            self._llm_status.setStyleSheet("color: #f44336; font-size: 11px;")

    def _download_model(self) -> None:
        """Download a GGUF model (placeholder — launches script)."""
        import subprocess
        try:
            subprocess.Popen(
                ["python", "scripts/download_llm.py"],
                cwd=str(Path(__file__).parent.parent.parent),
            )
            QMessageBox.information(
                self, "Download Started",
                "Model download script launched in terminal.\n\n"
                "This runs in a separate process. Check the terminal for progress."
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start download:\n{e}")

    def _test_voice(self) -> None:
        """Test the selected voice."""
        voice = self._voice_combo.currentText()
        QMessageBox.information(
            self, "Voice Test",
            f"Testing voice '{voice}'...\n"
            f"(Requires Piper TTS with the voice model installed)"
        )

    def _import_voice_model(self) -> None:
        """Import a voice model from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Model File", "",
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            target_dir = Path(__file__).parent.parent.parent / "models" / "voice"
            target_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(path, target_dir / Path(path).name)
            self._voice_status.setText(f"✓ Imported {Path(path).name}")
            self._voice_status.setStyleSheet("color: #4caf50; font-size: 11px;")
            logger.info(f"Voice model imported: {path}")

    def _download_voice_model(self) -> None:
        """Download a voice model in background."""
        voice = self._voice_combo.currentText()
        target_dir = Path(__file__).parent.parent.parent / "models" / "voice"

        self._voice_progress.setVisible(True)
        self._voice_progress.setValue(0)
        self._voice_status.setText(f"Downloading {voice}...")
        self._voice_status.setStyleSheet("color: #ffa500; font-size: 11px;")

        self._voice_downloader = VoiceModelDownloader(voice, target_dir)
        self._voice_downloader.progress.connect(self._voice_progress.setValue)
        self._voice_downloader.status.connect(self._voice_status.setText)
        self._voice_downloader.finished.connect(self._on_voice_downloaded)
        self._voice_downloader.start()

    def _on_voice_downloaded(self, success: bool, msg: str) -> None:
        """Handle voice download completion."""
        self._voice_progress.setVisible(False)
        if success:
            self._voice_status.setText(f"✓ {msg}")
            self._voice_status.setStyleSheet("color: #4caf50; font-size: 11px;")
        else:
            self._voice_status.setText(f"✗ {msg}")
            self._voice_status.setStyleSheet("color: #f44336; font-size: 11px;")

    def _refresh_specs(self) -> None:
        """Refresh system specifications."""
        self._system_specs = detect_system_specs()
        QMessageBox.information(
            self, "Refreshed",
            f"System specs refreshed.\n"
            f"RAM: {self._system_specs.total_ram_gb:.1f} GB\n"
            f"VRAM: {self._system_specs.vram_gb:.1f} GB"
        )

    def _clear_memory(self) -> None:
        """Clear long-term memory cache."""
        msg = QMessageBox.question(
            self, "Clear Memory",
            "Are you sure you want to clear all long-term memory records?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if msg == QMessageBox.StandardButton.Yes:
            env_path = Path(__file__).parent.parent.parent / ".env"
            with open(env_path, "a") as f:
                f.write(f"\n# Cleared at {__import__('datetime').datetime.now()}\n")
            self._mem_label.setText("Long-term memory records: 0 (cleared)")
            logger.info("Memory cache cleared by user")

    def _load_values(self) -> None:
        """Load saved config values into UI."""
        cfg = self._config

        # AI Settings
        temp = cfg.get("temperature", 0.7)
        self._temp_slider.setValue(int(temp * 100))
        self._temp_value.setText(f"{temp:.2f}")

        ctx = cfg.get("context_length", 2048)
        self._context_spin.setValue(ctx)

        persona = cfg.get("persona", PersonaType.WITTY_COMPANION.value)
        idx = self._persona_combo.findText(persona)
        if idx >= 0:
            self._persona_combo.setCurrentIndex(idx)

        # API Keys
        self._openai_key.setText(cfg.get("openai_api_key", ""))
        self._openai_model.setCurrentText(cfg.get("openai_model", "gpt-4o-mini"))
        self._anthropic_key.setText(cfg.get("anthropic_api_key", ""))
        self._anthropic_model.setCurrentText(cfg.get("anthropic_model", "claude-haiku-3-5-20241022"))
        self._hybrid_mode.setChecked(cfg.get("hybrid_mode", False))

        # LLM Path
        llm_path = cfg.get("llm_path", os.environ.get("BUBBY_LLM_PATH", ""))
        self._llm_path_edit.setText(llm_path)
        self._check_model()

        # Features
        self._tts_check.setChecked(cfg.get("tts_enabled", os.environ.get("BUBBY_USE_TTS", "0") == "1"))
        self._autonomy_check.setChecked(cfg.get("autonomy_enabled", True))
        self._vision_check.setChecked(cfg.get("vision_enabled", True))

        # Voice
        self._mute_check.setChecked(cfg.get("mute", False))
        self._volume_slider.setValue(cfg.get("volume", 80))
        self._speed_slider.setValue(cfg.get("speed", 100))
        voice = cfg.get("voice_model", "amy")
        idx = self._voice_combo.findText(voice)
        if idx >= 0:
            self._voice_combo.setCurrentIndex(idx)

        # Character
        skin = cfg.get("skin", "Default Slime 🫧")
        idx = self._skin_combo.findText(skin)
        if idx >= 0:
            self._skin_combo.setCurrentIndex(idx)
        size = cfg.get("window_size", 400)
        self._size_slider.setValue(size)
        self._size_value.setText(f"{size}px")

        sprite_style = cfg.get("sprite_style", "Emoji (built-in)")
        idx = self._sprite_combo.findText(sprite_style)
        if idx >= 0:
            self._sprite_combo.setCurrentIndex(idx)

        self._bob_check.setChecked(cfg.get("bob_animation", True))
        self._wander_check.setChecked(cfg.get("wander_enabled", True))
        self._popup_check.setChecked(cfg.get("popup_enabled", True))

        # Performance
        self._efficient_mode.setChecked(cfg.get("efficient_mode", self._system_specs.efficient_mode_recommended()))
        records = cfg.get("memory_records", "—")
        self._mem_label.setText(f"Long-term memory records: {records}")

    def _save_and_close(self) -> None:
        """Save all settings and close."""
        cfg = dict(self._config)

        # AI Settings
        cfg["temperature"] = self._temp_slider.value() / 100
        cfg["context_length"] = self._context_spin.value()
        cfg["persona"] = self._persona_combo.currentText()

        # API Keys
        if self._openai_key.text():
            cfg["openai_api_key"] = self._openai_key.text()
        if self._anthropic_key.text():
            cfg["anthropic_api_key"] = self._anthropic_key.text()
        cfg["openai_model"] = self._openai_model.currentText()
        cfg["anthropic_model"] = self._anthropic_model.currentText()
        cfg["hybrid_mode"] = self._hybrid_mode.isChecked()

        # LLM
        p = self._llm_path_edit.text().strip()
        if p:
            cfg["llm_path"] = p

        # Features
        cfg["tts_enabled"] = self._tts_check.isChecked()
        cfg["autonomy_enabled"] = self._autonomy_check.isChecked()
        cfg["vision_enabled"] = self._vision_check.isChecked()

        # Voice
        cfg["mute"] = self._mute_check.isChecked()
        cfg["volume"] = self._volume_slider.value()
        cfg["speed"] = self._speed_slider.value()
        cfg["voice_model"] = self._voice_combo.currentText()

        # Character
        cfg["skin"] = self._skin_combo.currentText()
        cfg["window_size"] = self._size_slider.value()
        cfg["sprite_style"] = self._sprite_combo.currentText()
        cfg["bob_animation"] = self._bob_check.isChecked()
        cfg["wander_enabled"] = self._wander_check.isChecked()
        cfg["popup_enabled"] = self._popup_check.isChecked()

        # Performance
        cfg["efficient_mode"] = self._efficient_mode.isChecked()

        save_config(cfg)

        # Write to .env for compatibility
        self._sync_to_env(cfg)

        logger.info("Settings saved to config.json and .env")
        self.settings_changed.emit()
        self.close()

    def _sync_to_env(self, cfg: dict) -> None:
        """Sync config values to .env file for legacy compatibility."""
        env_path = CONFIG_PATH.parent / ".env"
        env_lines = []
        if env_path.exists():
            env_lines = env_path.read_text().splitlines()

        new_lines = []
        found_keys = set()

        for line in env_lines:
            key = line.split("=")[0] if "=" in line else ""
            if key == "BUBBY_LLM_PATH":
                new_lines.append(f"BUBBY_LLM_PATH={cfg.get('llm_path', '')}")
                found_keys.add("BUBBY_LLM_PATH")
            elif key == "BUBBY_USE_TTS":
                val = "1" if cfg.get("tts_enabled", False) else "0"
                new_lines.append(f"BUBBY_USE_TTS={val}")
                found_keys.add("BUBBY_USE_TTS")
            elif key == "BUBBY_USE_LLM":
                val = "1" if cfg.get("llm_path", "") else "0"
                new_lines.append(f"BUBBY_USE_LLM={val}")
                found_keys.add("BUBBY_USE_LLM")
            elif key == "OPENAI_API_KEY":
                new_lines.append(f"OPENAI_API_KEY={cfg.get('openai_api_key', '')}")
                found_keys.add("OPENAI_API_KEY")
            elif key == "ANTHROPIC_API_KEY":
                new_lines.append(f"ANTHROPIC_API_KEY={cfg.get('anthropic_api_key', '')}")
                found_keys.add("ANTHROPIC_API_KEY")
            else:
                new_lines.append(line)

        # Add missing keys
        if "BUBBY_LLM_PATH" not in found_keys and cfg.get("llm_path"):
            new_lines.append(f"BUBBY_LLM_PATH={cfg['llm_path']}")
        if "BUBBY_USE_TTS" not in found_keys:
            new_lines.append(f"BUBBY_USE_TTS={'1' if cfg.get('tts_enabled', False) else '0'}")
        if "OPENAI_API_KEY" not in found_keys and cfg.get("openai_api_key"):
            new_lines.append(f"OPENAI_API_KEY={cfg['openai_api_key']}")
        if "ANTHROPIC_API_KEY" not in found_keys and cfg.get("anthropic_api_key"):
            new_lines.append(f"ANTHROPIC_API_KEY={cfg['anthropic_api_key']}")

        env_path.write_text("\n".join(new_lines) + "\n")

    def set_memory_records(self, count: int) -> None:
        """Update memory record count display."""
        self._mem_label.setText(f"Long-term memory records: {count}")

    def closeEvent(self, event) -> None:
        """Cleanup on close."""
        if self._voice_downloader and self._voice_downloader.isRunning():
            self._voice_downloader.quit()
            self._voice_downloader.wait(1000)
        super().closeEvent(event)