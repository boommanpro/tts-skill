<div align="center">

# tts-skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20·%20macOS%20·%20Windows-green.svg)](#requirements)
[![OmniVoice](https://img.shields.io/badge/Based%20on-OmniVoice-orange.svg)](https://github.com/k2-fsa/OmniVoice)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-blueviolet.svg)](https://github.com/SYSTRAN/faster-whisper)

**One command to make any text speak with any voice; one command to turn audio/video into text.**

Voice cloning, generation, and transcription tool based on open-source [OmniVoice](https://github.com/k2-fsa/OmniVoice) + [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Supports 600+ languages zero-shot TTS, voice cloning, voice design, speech-to-text (audio/video), cross-platform auto-install, and China mirror acceleration.

[Examples](#examples) · [Install](#install) · [Usage](#usage) · [Web UI](#web-ui) · [CLI](#cli) · [Docs](https://boommanpro.github.io/tts-skill/)

**Other Languages:**

[中文](README.md)

</div>

---

## Examples

### 1. Voice Cloning (Zero-shot)

```bash
# Upload a 3-10 second reference audio to clone its voice
python -m tts_skill infer \
  --text "The weather is nice today, let's go for a walk" \
  --mode clone \
  --ref_audio sample.wav \
  --ref_text "This is the transcript of the reference audio" \
  --seed 42
```

### 2. Voice Design (By Attributes)

```bash
# Describe voice attributes, model generates automatically
python -m tts_skill infer \
  --text "Hello, welcome to our service" \
  --mode design \
  --instruct "male, british accent, low pitch" \
  --seed 42
```

Supported voice attributes:

| Category | Options |
|----------|---------|
| Gender | Male / Female |
| Age | Child / Teenager / Young Adult / Middle-aged / Elderly |
| Pitch | Very Low / Low / Moderate / High / Very High Pitch |
| Style | Whisper |
| English Accent | American / British / Australian / Indian / Japanese / Korean ... |
| Chinese Dialect | Henan / Shaanxi / Sichuan / Northeastern / Guizhou ... |

### 3. Batch Generation (5 different seeds)

```bash
python -m tts_skill infer --text "Hello world" --mode auto --batch_count 5
```

Generates 5 audio files with different seeds, each independently reproducible.

### 4. Speech-to-Text (Audio/Video -> Text)

```bash
# Transcribe a video file (audio track auto-extracted)
python -m tts_skill transcribe --input meeting.mp4

# Transcribe an audio file with explicit language
python -m tts_skill transcribe --input audio.mp3 --language en

# Translate to English (original audio -> English subtitles)
python -m tts_skill transcribe --input video.mp4 --task translate

# Use a smaller model (for CPU or limited VRAM)
python -m tts_skill transcribe --input audio.mp3 --model small
```

Output:

```
Transcription complete!
Input file: meeting.mp4 (video)
Detected language: en (confidence 98.50%)
Audio duration: 1254.3s (after VAD 1180.5s)
Segments: 87
Model: large-v3 (device=cuda, task=transcribe)
Output files:
  - outputs/meeting_20260719_100000_a1b2c3d4.json  # with word-level timestamps
  - outputs/meeting_20260719_100000_a1b2c3d4.txt   # plain text

Reproducible command (copy to re-run with identical results):
------------------------------------------------------------
python -m tts_skill transcribe --input meeting.mp4 --model large-v3 --task transcribe --language en --device cuda
```

Supports video (mp4/mov/mkv/avi/webm/...) and audio (wav/mp3/flac/m4a/aac/ogg/...). Cross-platform ffmpeg is bundled, no manual installation required.

---

## Install

### Option 1: One-command install (Recommended)

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill
python -m tts_skill setup
```

Automatically detects platform and GPU, uses China mirror to install PyTorch + OmniVoice + faster-whisper (~10-20 minutes).

### Option 2: Run directly (auto-install)

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill

# Launch Web UI or CLI directly, environment auto-installs
python -m tts_skill web
# or
python -m tts_skill infer --text "Hello" --mode auto
```

### Cross-platform Support

| Platform | GPU Type | Selection |
|----------|----------|-----------|
| Linux | NVIDIA CUDA | torch+cu128 |
| Linux | Intel Arc | torch+xpu |
| Linux | No GPU | torch+cpu |
| macOS (Apple Silicon) | MPS | standard torch |
| macOS (Intel) | CPU | standard torch |
| Windows | NVIDIA CUDA | torch+cu128 |
| Windows | No GPU | torch+cpu |

### China Acceleration

China mirror enabled by default (disable via `TTS_SKILL_FORCE_REGION=global`):

- **pip mirror**: `https://mirrors.aliyun.com/pypi/simple`
- **HuggingFace mirror**: `https://hf-mirror.com`

---

## Usage

### Web UI

```bash
python -m tts_skill web
```

Visit `http://localhost:7860` in your browser. The UI has 5 tabs:

1. **Voice Clone** - Upload reference audio + input text
2. **Voice Design** - Select gender/age/pitch/accent/dialect
3. **Auto Voice** - Input text only
4. **Speech-to-Text** - Upload audio/video, auto-transcribe to text (with word-level timestamps)
5. **History** - View, rename, delete, view reproducible commands (TTS + ASR sections)

TTS tabs support:
- Random seed configuration (fixed for reproducibility / random for exploration)
- Batch generation (default 5, each with different seed, all displayed)
- Generation parameter tuning (num_step, guidance_scale, speed, duration, etc.)
- Reproducible CLI command output after generation

Speech-to-Text tab supports:
- Upload audio or video files (video audio track auto-extracted)
- Model selection (tiny/base/small/medium/large-v3, default large-v3)
- Auto language detection or manual specification
- Transcribe (keep source language) or translate to English
- Word-level timestamps, VAD silence filtering
- JSON output (with word-level timestamps) + plain text, directly downloadable in Web UI

### CLI

```bash
# === TTS Voice Generation ===
# Auto voice
python -m tts_skill infer --text "Hello" --mode auto --seed 42

# Voice cloning
python -m tts_skill infer --text "Hello" --mode clone --ref_audio ref.wav --seed 42

# Voice design
python -m tts_skill infer --text "Hello" --mode design --instruct "male, british accent" --seed 42

# Batch generate 5
python -m tts_skill infer --text "Hello" --mode auto --batch_count 5

# === Speech-to-Text ===
# Transcribe video (auto-extract audio track)
python -m tts_skill transcribe --input meeting.mp4

# Transcribe audio with explicit language
python -m tts_skill transcribe --input audio.mp3 --language en

# Translate to English
python -m tts_skill transcribe --input video.mp4 --task translate

# === History ===
# View all history (TTS + ASR)
python -m tts_skill history

# View only TTS / only ASR records
python -m tts_skill history --type tts
python -m tts_skill history --type asr

# Show reproducible commands for all history
python -m tts_skill history --show_cmd

# === Others ===
# Diagnose environment
python -m tts_skill doctor

# Show version
python -m tts_skill --version
```

### Python API

```python
from tts_skill.config import create_record, load_history, build_reproducible_command
from tts_skill.utils import fix_random_seed, gen_random_seed, detect_torch_device

# Set random seed for reproducibility
fix_random_seed(42)

# Detect device
device = detect_torch_device()  # 'cuda' / 'mps' / 'xpu' / 'cpu'

# Load history
records = load_history()
for r in records:
    print(r.name, r.seed, build_reproducible_command(r))
```

---

## Key Features

### 1. Reproducible Random Seed

Same seed + same parameters = identical audio output. Every generation records the seed and outputs a reproducible command.

### 2. Batch Generation

Generate multiple audio samples with different seeds at once (default 5) to explore the best result. All audio samples are displayed independently in Web UI and saved as separate files in CLI.

### 3. History Management

All TTS generation records are saved in `history.json`, ASR transcription records in `transcribe_history.json`. Supports:

- View all history (TTS + ASR)
- Edit record names
- Delete records (also deletes output files)
- View reproducible command for any record

### 4. Cross-platform Auto-install

One command automatically:
- Detects OS (Linux/macOS/Windows)
- Detects GPU type (CUDA/MPS/XPU/CPU)
- Selects correct PyTorch wheel
- Uses China mirror for download acceleration
- Installs OmniVoice, faster-whisper, and all dependencies

### 5. China-friendly

- pip mirror acceleration (Aliyun)
- HuggingFace mirror acceleration (hf-mirror.com)
- Enabled by default, can be disabled with `TTS_SKILL_FORCE_REGION=global`

---

## Repository Structure

```
tts-skill/
├── SKILL.md                  # Skill definition (AI Agent entry point)
├── README.md                 # Project README (Chinese)
├── README_EN.md              # Project README (English)
├── CONTRIBUTING.md           # Contributing guide
├── LICENSE                   # MIT License
├── pyproject.toml            # Python project config
├── requirements.txt          # Dependencies
│
├── tts_skill/                # Python package
│   ├── __main__.py           # Entry: python -m tts_skill
│   ├── cli.py                # Main CLI (setup/web/infer/transcribe/history/doctor)
│   ├── setup_env.py          # Cross-platform auto-install (TTS + ASR deps)
│   ├── webui.py              # Gradio Web UI (5 tabs)
│   ├── infer.py              # TTS inference
│   ├── transcribe.py         # Speech-to-text (faster-whisper)
│   ├── config.py             # TTS history management
│   └── utils.py              # Platform/device/seed utilities
│
├── tests/                    # Test cases (234 tests, all passing)
│   ├── test_utils.py         # Platform detection, seed, paths
│   ├── test_config.py        # History CRUD, reproducible commands
│   ├── test_setup_env.py     # Install strategy
│   ├── test_infer.py         # TTS inference flow
│   ├── test_transcribe.py    # Speech-to-text flow (61 tests)
│   ├── test_webui.py         # Web UI construction
│   └── test_cli.py           # CLI dispatch
│
├── docs/                     # GitHub Pages source
│   ├── index.md              # Docs home
│   └── _config.yml           # Jekyll config
│
└── .github/
    └── workflows/
        ├── deploy-pages.yml  # GitHub Pages deployment
        └── test.yml          # CI tests
```

---

## Requirements

- Python >= 3.10
- Internet connection for first run (downloads PyTorch + models, ~5GB)
- GPU recommended (NVIDIA CUDA / Apple Silicon MPS / Intel Arc XPU), CPU also works but slower

---

## Troubleshooting

Run `python -m tts_skill doctor` to diagnose environment issues.

Common issues:

- **Slow install**: China mirror enabled by default. Set `TTS_SKILL_FORCE_REGION=cn` to force enable
- **Model download failed**: `HF_ENDPOINT=https://hf-mirror.com` is set. Set manually if needed
- **GPU not detected**: Run doctor to check, or specify via `--device` (cuda/mps/xpu/cpu)
- **Overseas network**: Set `TTS_SKILL_FORCE_REGION=global` to disable China mirror
- **Port in use**: `python -m tts_skill web --port 7861` to specify another port
- **transcribe command unavailable**: Run `python -m tts_skill setup` to install ASR deps (faster-whisper + imageio-ffmpeg)
- **Slow transcription on macOS**: faster-whisper does not support Apple MPS, automatically falls back to CPU + int8. Use `--model small` to speed up
- **Video transcription fails**: imageio-ffmpeg bundles a cross-platform ffmpeg binary. If still failing, install system ffmpeg and set `FFMPEG_BINARY` env var

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — Use freely, modify freely, build freely.

Based on [OmniVoice](https://github.com/k2-fsa/OmniVoice) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Thanks to the k2-fsa and SYSTRAN teams for their open-source contributions.

MIT License © [boommanpro](https://github.com/boommanpro)
