<div align="center">

# tts-skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20·%20macOS%20·%20Windows-green.svg)](#requirements)
[![OmniVoice](https://img.shields.io/badge/Based%20on-OmniVoice-orange.svg)](https://github.com/k2-fsa/OmniVoice)

**One command to make any text speak with any voice.**

Voice cloning and generation tool based on open-source [OmniVoice](https://github.com/k2-fsa/OmniVoice). Supports 600+ languages zero-shot TTS, voice cloning, voice design, cross-platform auto-install, and China mirror acceleration.

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

### 3. Batch Generation (5 different seeds)

```bash
python -m tts_skill infer --text "Hello world" --mode auto --batch_count 5
```

---

## Install

### Option 1: One-command install (Recommended)

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill
python -m tts_skill setup
```

Automatically detects platform and GPU, uses China mirror to install PyTorch + OmniVoice (~10-20 minutes).

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

---

## Usage

### Web UI

```bash
python -m tts_skill web
```

Visit `http://localhost:7860` in your browser. The UI has 4 tabs:

1. **Voice Clone** - Upload reference audio + input text
2. **Voice Design** - Select gender/age/pitch/accent/dialect
3. **Auto Voice** - Input text only
4. **History** - View, rename, delete, view reproducible commands

Each tab supports:
- Random seed configuration (fixed for reproducibility / random for exploration)
- Batch generation (default 5, each with different seed, all displayed)
- Generation parameter tuning (num_step, guidance_scale, speed, duration, etc.)
- Reproducible CLI command output after generation

### CLI

```bash
# Auto voice
python -m tts_skill infer --text "Hello" --mode auto --seed 42

# Voice cloning
python -m tts_skill infer --text "Hello" --mode clone --ref_audio ref.wav --seed 42

# Voice design
python -m tts_skill infer --text "Hello" --mode design --instruct "male, british accent" --seed 42

# Batch generate 5
python -m tts_skill infer --text "Hello" --mode auto --batch_count 5

# View history
python -m tts_skill history

# Show reproducible commands for all history
python -m tts_skill history --show_cmd

# Diagnose environment
python -m tts_skill doctor

# Show version
python -m tts_skill --version
```

---

## Key Features

### 1. Reproducible Random Seed

Same seed + same parameters = identical audio output. Every generation records the seed and outputs a reproducible command.

### 2. Batch Generation

Generate multiple audio samples with different seeds at once (default 5) to explore the best result. All audio samples are displayed independently in Web UI and saved as separate files in CLI.

### 3. History Management

All generation records are saved in `history.json`, supporting:
- View all history
- Edit record names
- Delete records (also deletes output files)
- View reproducible command for any record

### 4. Cross-platform Auto-install

One command automatically:
- Detects OS (Linux/macOS/Windows)
- Detects GPU type (CUDA/MPS/XPU/CPU)
- Selects correct PyTorch wheel
- Uses China mirror for download acceleration
- Installs OmniVoice and all dependencies

### 5. China-friendly

- pip mirror acceleration (Aliyun)
- HuggingFace mirror acceleration (hf-mirror.com)
- Enabled by default, can be disabled with `TTS_SKILL_FORCE_REGION=global`

---

## Requirements

- Python >= 3.10
- Internet connection for first run (downloads PyTorch + model, ~5GB)
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

---

## License

MIT — Use freely, modify freely, build freely.

Based on [OmniVoice](https://github.com/k2-fsa/OmniVoice). Thanks to the k2-fsa team for the open-source contribution.

MIT License © [boommanpro](https://github.com/boommanpro)
