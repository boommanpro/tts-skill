<div align="center">

# tts-skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20·%20macOS%20·%20Windows-green.svg)](#环境要求)
[![OmniVoice](https://img.shields.io/badge/Based%20on-OmniVoice-orange.svg)](https://github.com/k2-fsa/OmniVoice)
[![China Mirror](https://img.shields.io/badge/China%20Mirror-Enabled-red.svg)](#国内加速)

**一行命令，让文本发出任何人的声音。**

基于开源 [OmniVoice](https://github.com/k2-fsa/OmniVoice) 的声音克隆与生成工具。支持 600+ 语种零样本文本转语音、声音克隆、声音设计，跨平台自动安装，国内镜像加速。

[效果示例](#效果示例) · [安装](#安装) · [使用](#使用) · [Web UI](#web-ui) · [CLI](#cli-命令) · [文档](https://boommanpro.github.io/tts-skill/)

**其他语言 / Other Languages:**

[English](README_EN.md)

</div>

---

## 效果示例

### 1. 声音克隆（零样本）

```bash
# 上传一段 3-10 秒的参考音频，克隆其声音
python -m tts_skill infer \
  --text "今天天气真不错，我们一起出去走走吧" \
  --mode clone \
  --ref_audio sample.wav \
  --ref_text "这是一段参考音频的文本" \
  --seed 42
```

输出：

```
[1/1] 生成中 (seed=42, mode=clone)...
[1/1] 已保存: outputs/clone_20260719_100000_seed42.wav

============================================================
生成完成！
============================================================
记录 ID: a1b2c3d4
基础种子: 42
输出文件:
  - outputs/clone_20260719_100000_seed42.wav

可复现命令（复制以下命令可重新生成相同结果）:
------------------------------------------------------------
/Users/user/.venv/bin/python -m tts_skill infer --text "今天天气真不错..." --mode clone --seed 42 --ref_audio sample.wav ...
```

### 2. 声音设计（按属性生成）

```bash
# 描述声音属性，模型自动生成
python -m tts_skill infer \
  --text "Hello, welcome to our service" \
  --mode design \
  --instruct "male, british accent, low pitch" \
  --seed 42
```

支持的声音属性：

| 类别 | 选项 |
|------|------|
| 性别 | Male / Female |
| 年龄 | Child / Teenager / Young Adult / Middle-aged / Elderly |
| 音调 | Very Low / Low / Moderate / High / Very High Pitch |
| 风格 | Whisper |
| 英文口音 | American / British / Australian / Indian / Japanese / Korean ... |
| 中文方言 | 河南话 / 陕西话 / 四川话 / 东北话 / 贵州话 ... |

### 3. 批量生成（5 个不同种子）

```bash
python -m tts_skill infer --text "你好世界" --mode auto --batch_count 5
```

生成 5 个不同种子的音频文件，每个都可独立复现。

### 4. 语音转文字（音频/视频 -> 文本）

```bash
# 转写视频文件（自动提取音轨）
python -m tts_skill transcribe --input meeting.mp4

# 转写音频文件，指定语言
python -m tts_skill transcribe --input audio.mp3 --language zh

# 翻译成英文（视频原声 -> 英文字幕）
python -m tts_skill transcribe --input video.mp4 --task translate

# 使用更小的模型（CPU 或显存不足时）
python -m tts_skill transcribe --input audio.mp3 --model small
```

输出：

```
转写完成！
输入文件: meeting.mp4 (video)
检测语言: zh (置信度 98.50%)
音频时长: 1254.3s (VAD 后 1180.5s)
分段数量: 87
模型: large-v3 (device=cuda, task=transcribe)
输出文件:
  - outputs/meeting_20260719_100000_a1b2c3d4.json  # 含词级时间戳
  - outputs/meeting_20260719_100000_a1b2c3d4.txt   # 纯文本

可复现命令（复制以下命令可重新转写相同结果）:
------------------------------------------------------------
python -m tts_skill transcribe --input meeting.mp4 --model large-v3 --task transcribe --language zh --device cuda
```

支持视频（mp4/mov/mkv/avi/webm/...）和音频（wav/mp3/flac/m4a/aac/ogg/...），跨平台 ffmpeg 自带，无需用户手动安装。

---

## 安装

### 方式一：一键安装（推荐）

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill
python -m tts_skill setup
```

首次运行会自动检测平台和 GPU，使用国内镜像安装 PyTorch + OmniVoice（约 10-20 分钟）。

### 方式二：直接运行（自动安装）

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill

# 直接启动 Web UI 或 CLI，会自动触发环境安装
python -m tts_skill web
# 或
python -m tts_skill infer --text "你好" --mode auto
```

### 跨平台支持

| 平台 | GPU 类型 | 自动选择 |
|------|----------|----------|
| Linux | NVIDIA CUDA | torch+cu128 |
| Linux | Intel Arc | torch+xpu |
| Linux | 无 GPU | torch+cpu |
| macOS (Apple Silicon) | MPS | 标准 torch |
| macOS (Intel) | CPU | 标准 torch |
| Windows | NVIDIA CUDA | torch+cu128 |
| Windows | 无 GPU | torch+cpu |

### 国内加速

默认启用国内镜像（可通过 `TTS_SKILL_FORCE_REGION=global` 关闭）：

- **pip 镜像**：`https://mirrors.aliyun.com/pypi/simple`
- **HuggingFace 镜像**：`https://hf-mirror.com`

---

## 使用

### Web UI

```bash
python -m tts_skill web
```

浏览器访问 `http://localhost:7860`，界面包含 5 个标签页：

1. **声音克隆** - 上传参考音频 + 输入文本
2. **声音设计** - 选择性别/年龄/音调/口音/方言
3. **自动声音** - 仅输入文本
4. **语音转文字** - 上传音频/视频，自动转写为文字（含词级时间戳）
5. **历史记录** - 查看、重命名、删除、查看复现命令（TTS + ASR 分区）

TTS 标签页支持：

- 随机种子配置（固定可复现 / 随机探索）
- 批量生成（默认 5 个，每个不同种子，全部在界面展示）
- 生成参数调节（num_step、guidance_scale、speed、duration 等）
- 生成后输出可复现 CLI 命令

语音转文字标签页支持：

- 上传音频或视频文件（视频自动提取音轨）
- 模型选择（tiny/base/small/medium/large-v3，默认 large-v3）
- 语言自动检测或手动指定
- 转写（保留原语言）或翻译成英文
- 词级时间戳、VAD 静音过滤
- 输出 JSON（含词级时间戳）+ 纯文本，Web UI 直接下载

### CLI 命令

```bash
# === TTS 声音生成 ===
# 自动声音
python -m tts_skill infer --text "你好" --mode auto --seed 42

# 声音克隆
python -m tts_skill infer --text "你好" --mode clone --ref_audio ref.wav --seed 42

# 声音设计
python -m tts_skill infer --text "Hello" --mode design --instruct "male, british accent" --seed 42

# 批量生成 5 个
python -m tts_skill infer --text "你好" --mode auto --batch_count 5

# === 语音转文字 ===
# 转写视频（自动提取音轨）
python -m tts_skill transcribe --input meeting.mp4

# 转写音频并指定语言
python -m tts_skill transcribe --input audio.mp3 --language zh

# 翻译成英文
python -m tts_skill transcribe --input video.mp4 --task translate

# === 历史记录 ===
# 查看所有历史记录（TTS + ASR）
python -m tts_skill history

# 只看 TTS 记录 / 只看转写记录
python -m tts_skill history --type tts
python -m tts_skill history --type asr

# 显示所有历史记录的复现命令
python -m tts_skill history --show_cmd

# === 其他 ===
# 诊断环境问题
python -m tts_skill doctor

# 查看版本
python -m tts_skill --version
```

### Python API

```python
from tts_skill.config import create_record, load_history, build_reproducible_command
from tts_skill.utils import fix_random_seed, gen_random_seed, detect_torch_device

# 设置随机种子保证可复现
fix_random_seed(42)

# 检测设备
device = detect_torch_device()  # 'cuda' / 'mps' / 'xpu' / 'cpu'

# 加载历史记录
records = load_history()
for r in records:
    print(r.name, r.seed, build_reproducible_command(r))
```

---

## 核心特性

### 1. 随机种子可复现

相同种子 + 相同参数 = 完全相同的音频输出。每次生成都会记录种子，并输出可复现命令。

### 2. 批量生成

一次生成多个不同种子的音频（默认 5 个），用于探索最佳效果。所有音频在 Web UI 中独立展示，在 CLI 中独立保存。

### 3. 历史记录管理

所有生成记录保存在 `history.json`，支持：

- 查看所有历史记录
- 编辑记录名称
- 删除记录（同时删除输出文件）
- 查看任意记录的可复现命令

### 4. 跨平台自动安装

一行命令自动完成：

- 检测操作系统（Linux/macOS/Windows）
- 检测 GPU 类型（CUDA/MPS/XPU/CPU）
- 选择正确的 PyTorch wheel
- 使用国内镜像加速下载
- 安装 OmniVoice 及所有依赖

### 5. 国内友好

- pip 镜像加速（阿里云）
- HuggingFace 镜像加速（hf-mirror.com）
- 默认启用，海外用户可关闭

---

## 仓库结构

```
tts-skill/
├── SKILL.md                  # Skill 定义（AI Agent 调用入口）
├── README.md                 # 项目说明（中文）
├── README_EN.md              # 项目说明（英文）
├── CONTRIBUTING.md           # 贡献指南
├── LICENSE                   # MIT 许可证
├── pyproject.toml            # Python 项目配置
├── requirements.txt          # 依赖声明
│
├── tts_skill/                # Python 包
│   ├── __main__.py           # 入口：python -m tts_skill
│   ├── cli.py                # 主 CLI（setup/web/infer/transcribe/history/doctor）
│   ├── setup_env.py          # 跨平台自动安装（TTS + ASR 依赖）
│   ├── webui.py              # Gradio Web UI（5 个 Tab）
│   ├── infer.py              # TTS 推理
│   ├── transcribe.py         # 语音转文字（faster-whisper）
│   ├── config.py             # TTS 历史记录管理
│   └── utils.py              # 平台/设备/种子工具
│
├── tests/                    # 测试用例（234 个，全部通过）
│   ├── test_utils.py         # 平台检测、种子、路径
│   ├── test_config.py        # 历史 CRUD、复现命令
│   ├── test_setup_env.py     # 安装策略
│   ├── test_infer.py         # TTS 推理流程
│   ├── test_transcribe.py    # 语音转文字流程（61 个）
│   ├── test_webui.py         # Web UI 构建
│   └── test_cli.py           # CLI 分发
│
├── docs/                     # GitHub Pages 文档源
│   ├── index.md              # 文档首页
│   └── _config.yml           # Jekyll 配置
│
└── .github/
    └── workflows/
        ├── deploy-pages.yml  # GitHub Pages 部署
        └── test.yml          # CI 测试
```

---

## 环境要求

- Python >= 3.10
- 首次运行需联网（下载 PyTorch + 模型，约 5GB）
- 推荐有 GPU（NVIDIA CUDA / Apple Silicon MPS / Intel Arc XPU），CPU 也可运行但较慢

---

## 故障排查

运行 `python -m tts_skill doctor` 诊断环境问题。

常见问题：

- **安装慢**：脚本已默认启用国内镜像，如仍慢可设置 `TTS_SKILL_FORCE_REGION=cn`
- **模型下载失败**：已设置 `HF_ENDPOINT=https://hf-mirror.com`，如失败可手动设置
- **GPU 未识别**：运行 doctor 检查，或通过 `--device` 手动指定（cuda/mps/xpu/cpu）
- **海外网络**：设置 `TTS_SKILL_FORCE_REGION=global` 关闭国内镜像
- **端口被占用**：`python -m tts_skill web --port 7861` 指定其他端口

---

## 贡献

欢迎提 Issue 和 PR。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MIT — 随便用，随便改，随便造。

基于 [OmniVoice](https://github.com/k2-fsa/OmniVoice) 项目开发，感谢 k2-fsa 团队的开源贡献。

MIT License © [boommanpro](https://github.com/boommanpro)
