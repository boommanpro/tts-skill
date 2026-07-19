# tts-skill

> 一行命令，让文本发出任何人的声音；一行命令，让音频/视频变成文字。

基于开源 [OmniVoice](https://github.com/k2-fsa/OmniVoice) + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的声音克隆/生成/转写工具。支持 600+ 语种零样本文本转语音、声音克隆、声音设计、语音转文字（音频/视频），跨平台自动安装，国内镜像加速。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/boommanpro/tts-skill/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20·%20macOS%20·%20Windows-green.svg)](https://github.com/boommanpro/tts-skill#环境要求)
[![OmniVoice](https://img.shields.io/badge/Based%20on-OmniVoice-orange.svg)](https://github.com/k2-fsa/OmniVoice)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-blueviolet.svg)](https://github.com/SYSTRAN/faster-whisper)

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/boommanpro/tts-skill.git
cd tts-skill
```

### 2. 启动 Web UI（推荐）

```bash
python -m tts_skill web
```

首次运行会自动安装环境（约 10-20 分钟，国内镜像加速）。安装完成后浏览器访问 `http://localhost:7860`。

### 3. CLI 直接使用

```bash
# === 声音生成（TTS）===
# 自动声音
python -m tts_skill infer --text "你好，这是一个测试" --mode auto --seed 42

# 声音克隆
python -m tts_skill infer --text "你好" --mode clone --ref_audio ref.wav --seed 42

# 声音设计
python -m tts_skill infer --text "Hello" --mode design --instruct "male, british accent" --seed 42

# 批量生成 5 个
python -m tts_skill infer --text "你好" --mode auto --batch_count 5

# === 语音转文字（ASR）===
# 转写视频（自动提取音轨）
python -m tts_skill transcribe --input meeting.mp4

# 转写音频
python -m tts_skill transcribe --input audio.mp3 --language zh

# 翻译成英文
python -m tts_skill transcribe --input video.mp4 --task translate
```

---

## 核心功能

### 声音克隆（零样本）

上传一段 3-10 秒的参考音频，克隆其声音生成任意文本的语音。

```bash
python -m tts_skill infer \
  --text "今天天气真不错" \
  --mode clone \
  --ref_audio sample.wav \
  --ref_text "这是参考音频的文本" \
  --seed 42
```

### 声音设计（按属性生成）

通过描述声音属性（性别、年龄、音调、口音、方言）生成定制声音。

| 类别 | 选项 |
|------|------|
| 性别 | Male / Female |
| 年龄 | Child / Teenager / Young Adult / Middle-aged / Elderly |
| 音调 | Very Low / Low / Moderate / High / Very High Pitch |
| 风格 | Whisper |
| 英文口音 | American / British / Australian / Indian / Japanese / Korean ... |
| 中文方言 | 河南话 / 陕西话 / 四川话 / 东北话 / 贵州话 ... |

### 自动声音

仅输入文本，模型自动选择合适的声音。

```bash
python -m tts_skill infer --text "你好世界" --mode auto --seed 42
```

---

## 随机种子与可复现

**相同种子 + 相同参数 = 完全相同的音频输出。**

每次生成都会：

1. 记录种子到历史记录
2. 输出可复现 CLI 命令
3. 保存完整参数到 `history.json`

```
============================================================
生成完成！
============================================================
记录 ID: a1b2c3d4
基础种子: 42
输出文件:
  - outputs/auto_20260719_100000_seed42.wav

可复现命令（复制以下命令可重新生成相同结果）:
------------------------------------------------------------
python -m tts_skill infer --text "你好" --mode auto --seed 42 ...
```

### 批量生成

一次生成多个不同种子的音频（默认 5 个），用于探索最佳效果：

```bash
python -m tts_skill infer --text "你好" --mode auto --batch_count 5
```

会生成 5 个文件，种子分别为 `base_seed + 0, 1, 2, 3, 4`。

---

## Web UI

启动 Web UI：

```bash
python -m tts_skill web
```

浏览器访问 `http://localhost:7860`，包含 5 个标签页：

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

---

## CLI 命令参考

### infer - 生成音频

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--text` | 要合成的文本（必填） | - |
| `--mode` | 生成模式：clone/design/auto | auto |
| `--ref_audio` | 参考音频路径（clone 模式必填） | - |
| `--ref_text` | 参考音频文本（可选） | - |
| `--instruct` | 声音属性（design 模式） | - |
| `--language` | 语种（如 Chinese, en） | 自动检测 |
| `--seed` | 随机种子 | 随机 |
| `--batch_count` | 批量生成数量 | 1 |
| `--num_step` | 解码步数 | 32 |
| `--speed` | 语速 | 1.0 |
| `--duration` | 固定时长（秒） | - |
| `--guidance_scale` | 引导尺度 | 2.0 |
| `--denoise` | 是否降噪 | true |
| `--normalize_text` | 文本归一化 | false |

### transcribe - 语音转文字

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入音频/视频文件路径（必填） | - |
| `--model` | Whisper 模型：tiny/base/small/medium/large-v3 | large-v3 |
| `--language` | 语言代码（如 zh、en、ja），不指定则自动检测 | 自动检测 |
| `--task` | 任务：transcribe（转写）/ translate（翻译成英文） | transcribe |
| `--device` | 设备：cuda / cpu（不支持 MPS，macOS 自动回退 cpu） | 自动检测 |
| `--compute_type` | 计算精度：float16 / int8 / ... | cuda→float16, cpu→int8 |
| `--beam_size` | beam search 大小 | 5 |
| `--word_timestamps` | 输出词级时间戳 | true |
| `--vad_filter` | VAD 静音过滤 | true |
| `--name` | 自定义记录名称 | - |
| `--output_dir` | 输出目录 | outputs/ |
| `--keep_temp_audio` | 保留从视频提取的临时音频文件 | false |

支持的视频格式：mp4 / mov / mkv / avi / webm / flv / wmv / m4v / mpg / mpeg / ts / 3gp / ogv

支持的音频格式：wav / mp3 / flac / m4a / aac / ogg / wma / aiff / opus / oga

### 其他命令

```bash
python -m tts_skill setup          # 安装环境
python -m tts_skill setup --force  # 强制重新安装
python -m tts_skill setup --skip_asr  # 跳过 ASR 依赖（仅装 TTS）
python -m tts_skill history            # 查看所有历史记录（TTS + ASR）
python -m tts_skill history --type tts # 仅查看 TTS 记录
python -m tts_skill history --type asr # 仅查看转写记录
python -m tts_skill history --show_cmd # 显示复现命令
python -m tts_skill doctor         # 诊断环境
python -m tts_skill --version      # 查看版本
```

---

## 跨平台支持

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

默认启用国内镜像：

- **pip 镜像**：`https://mirrors.aliyun.com/pypi/simple`
- **HuggingFace 镜像**：`https://hf-mirror.com`

关闭国内镜像（海外用户）：

```bash
export TTS_SKILL_FORCE_REGION=global
```

---

## 故障排查

运行诊断命令：

```bash
python -m tts_skill doctor
```

输出包含：平台信息、Python 版本、网络区域、设备检测、安装状态、torch/omnivoice/gradio 版本、faster-whisper 与 ffmpeg 路径等。

常见问题：

- **安装慢**：脚本已默认启用国内镜像
- **模型下载失败**：检查 `HF_ENDPOINT` 环境变量
- **GPU 未识别**：运行 doctor 检查，或 `--device` 手动指定
- **端口被占用**：`--port` 指定其他端口
- **transcribe 命令不可用**：运行 `python -m tts_skill setup` 安装 ASR 依赖（faster-whisper + imageio-ffmpeg）
- **macOS 转写慢**：faster-whisper 不支持 Apple MPS，自动回退到 CPU + int8；可改用 `--model small` 加速
- **视频转写失败**：imageio-ffmpeg 自带跨平台 ffmpeg 二进制；如仍失败可手动安装系统 ffmpeg 并设置 `FFMPEG_BINARY` 环境变量

---

## 链接

- **GitHub 仓库**：[boommanpro/tts-skill](https://github.com/boommanpro/tts-skill)
- **基于项目**：[OmniVoice](https://github.com/k2-fsa/OmniVoice) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **问题反馈**：[Issues](https://github.com/boommanpro/tts-skill/issues)

## 许可证

MIT License © [boommanpro](https://github.com/boommanpro)
