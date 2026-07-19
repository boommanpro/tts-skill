---
name: tts-skill
description: |
  OmniVoice + faster-whisper 声音克隆/生成/转写工具：输入文本/参考音频/声音属性，自动完成 600+ 语种的零样本文本转语音、声音克隆、声音设计、批量生成；输入音频或视频文件，自动转写为带词级时间戳的文字。
  四种入口：(1)Web UI 可视化调参 (2)CLI 一行命令生成 TTS (3)CLI 一行命令转写音频/视频 (4)批量生成+可复现命令输出。
  触发词：「文本转语音」「声音克隆」「语音合成」「生成语音」「TTS」「调参生成」「批量生成音频」「声音设计」「复现语音」「语音转文字」「转写」「字幕」「提取文字」「视频转文字」「音频转文字」「ASR」「Whisper」。
  English triggers: "text to speech", "voice cloning", "speech synthesis", "generate audio", "TTS", "voice design", "batch generate audio", "speech to text", "transcribe", "transcription", "ASR", "subtitles", "video to text", "audio to text", "whisper".
---

# OmniVoice TTS Skill

> 「一行命令，让文本发出任何人的声音；一行命令，让音频/视频变成文字。」

基于开源项目 [OmniVoice](https://github.com/k2-fsa/OmniVoice) + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的声音克隆/生成/转写工具。支持 600+ 语种零样本文本转语音、音频/视频转文字，跨平台（Linux/macOS/Windows）自动安装，国内镜像加速。

## 何时调用此 Skill

当用户有以下需求时调用：

- 文本转语音（TTS）/ 语音合成 / 生成语音
- 声音克隆（用参考音频复刻声音）
- 声音设计（按性别、年龄、音调、口音等属性生成）
- 批量生成音频
- 需要可视化调参后生成语音
- 需要可复现的语音生成（随机种子）
- 语音转文字 / 转写音频 / 转写视频 / 提取字幕
- 视频或音频文件需要变成文字

## 一键执行（无需思考，直接运行）

### 1. 启动 Web UI（可视化调参，推荐）

```bash
python -m tts_skill web
```

首次运行会自动安装环境（约 10-20 分钟，国内镜像加速）。安装完成后浏览器访问 `http://localhost:7860`。

Web UI 功能（5 个标签页）：

- **声音克隆**：上传参考音频 + 输入文本 → 克隆声音
- **声音设计**：选择性别/年龄/音调/口音/方言 → 生成声音
- **自动声音**：仅输入文本 → 模型自动选择声音
- **语音转文字**：上传音频/视频 → 自动转写为文字（含词级时间戳，支持翻译成英文）
- **历史记录**：查看、重命名、删除、查看复现命令（TTS + ASR 分区）

每个 TTS 标签页还支持：

- **随机种子配置**：固定种子可复现，随机种子可探索
- **批量生成**：默认 5 个，每个不同种子，全部在界面展示

### 2. CLI 直接生成（无需 Web）

```bash
# 自动声音
python -m tts_skill infer --text "你好，这是一个测试" --mode auto --seed 42

# 声音克隆
python -m tts_skill infer --text "你好" --mode clone --ref_audio ref.wav --ref_text "参考文本" --seed 42

# 声音设计
python -m tts_skill infer --text "Hello" --mode design --instruct "male, british accent" --seed 42

# 批量生成 5 个
python -m tts_skill infer --text "你好" --mode auto --batch_count 5
```

生成完成后会输出**可复现命令**，复制即可重新生成相同结果。

### 3. CLI 转写音频/视频

```bash
# 转写视频（自动提取音轨）
python -m tts_skill transcribe --input meeting.mp4

# 转写音频，指定语言
python -m tts_skill transcribe --input audio.mp3 --language zh

# 翻译成英文（视频原声 -> 英文字幕）
python -m tts_skill transcribe --input video.mp4 --task translate

# 使用更小的模型（CPU 或显存不足时加速）
python -m tts_skill transcribe --input audio.mp3 --model small
```

支持视频（mp4/mov/mkv/avi/webm/...）和音频（wav/mp3/flac/m4a/aac/ogg/...），跨平台 ffmpeg 自带，无需用户手动安装。转写完成后输出 JSON（含词级时间戳）+ 纯文本，并展示**可复现命令**。

### 4. 其他命令

```bash
python -m tts_skill setup          # 单独安装环境
python -m tts_skill setup --force  # 强制重新安装
python -m tts_skill setup --skip_asr  # 跳过 ASR 依赖（仅装 TTS）
python -m tts_skill history            # 查看所有历史记录（TTS + ASR）
python -m tts_skill history --type tts # 仅查看 TTS 记录
python -m tts_skill history --type asr # 仅查看转写记录
python -m tts_skill history --show_cmd  # 显示所有复现命令
python -m tts_skill doctor         # 诊断环境问题
python -m tts_skill --version      # 查看版本
```

## 执行规则（AI Agent 必读）

1. **无需思考，直接运行命令**：所有逻辑已封装在 Python 脚本中
2. **首次使用先 setup**：如果用户首次使用，先运行 `python -m tts_skill setup`（或直接运行 web/infer/transcribe，会自动安装）
3. **国内环境自动加速**：脚本默认启用国内 pip 镜像和 HuggingFace 镜像（`hf-mirror.com`）
4. **跨平台自动适配**：自动检测 Linux/macOS/Windows 和 GPU 类型（CUDA/MPS/XPU/CPU）
5. **Web 优先**：用户需要调参时，启动 Web UI；用户明确要 CLI 时，用 infer/transcribe 命令
6. **复现命令**：生成/转写后务必展示给用户的可复现命令，方便后续复现
7. **ASR 依赖可选**：若仅用 TTS，可 `setup --skip_asr` 跳过 ASR 依赖；转写功能需 faster-whisper + imageio-ffmpeg
8. **macOS 转写注意**：faster-whisper 不支持 Apple MPS，自动回退 CPU + int8；如需加速可 `--model small`

## 参数说明

### infer 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--text` | 要合成的文本（必填） | - |
| `--mode` | 生成模式：clone/design/auto | auto |
| `--ref_audio` | 参考音频路径（clone 模式必填） | - |
| `--ref_text` | 参考音频文本（可选，留空自动转写） | - |
| `--instruct` | 声音属性（design 模式，如 "male, british accent"） | - |
| `--language` | 语种（如 Chinese, en） | 自动检测 |
| `--seed` | 随机种子（相同种子可复现） | 随机 |
| `--batch_count` | 批量生成数量 | 1 |
| `--num_step` | 解码步数（更小更快） | 32 |
| `--speed` | 语速（>1 更快） | 1.0 |
| `--duration` | 固定时长（秒，覆盖 speed） | - |
| `--guidance_scale` | 引导尺度 | 2.0 |
| `--denoise` | 是否降噪 | true |
| `--normalize_text` | 文本归一化 | false |

### 声音设计属性（instruct）

- **性别**：male, female / 男, 女
- **年龄**：child, teenager, young adult, middle-aged, elderly / 儿童, 少年, 青年, 中年, 老年
- **音调**：very low pitch, low pitch, moderate pitch, high pitch, very high pitch
- **风格**：whisper / 耳语
- **英文口音**：american accent, british accent, australian accent, indian accent 等
- **中文方言**：河南话, 陕西话, 四川话, 东北话 等

### transcribe 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入音频/视频文件路径（必填） | - |
| `--model` | Whisper 模型：tiny/base/small/medium/large-v3 | large-v3 |
| `--language` | 语言代码（如 zh、en、ja），不指定则自动检测 | 自动检测 |
| `--task` | 任务：transcribe（转写）/ translate（翻译成英文） | transcribe |
| `--device` | 设备：cuda / cpu（不支持 MPS，macOS 自动回退 cpu） | 自动检测 |
| `--compute_type` | 计算精度：float16 / int8 | cuda→float16, cpu→int8 |
| `--beam_size` | beam search 大小 | 5 |
| `--word_timestamps` | 输出词级时间戳 | true |
| `--vad_filter` | VAD 静音过滤 | true |
| `--name` | 自定义记录名称 | - |
| `--output_dir` | 输出目录 | outputs/ |
| `--keep_temp_audio` | 保留从视频提取的临时音频文件 | false |

## 环境要求

- Python >= 3.10
- 首次运行需联网（下载 PyTorch + 模型，约 5GB）
- 推荐有 GPU（NVIDIA CUDA / Apple Silicon MPS / Intel Arc XPU），CPU 也可运行但较慢

## 故障排查

运行 `python -m tts_skill doctor` 诊断环境问题。

常见问题：

- **安装慢**：脚本已默认启用国内镜像，如仍慢可设置 `TTS_SKILL_FORCE_REGION=cn`
- **模型下载失败**：已设置 `HF_ENDPOINT=https://hf-mirror.com`，如失败可手动设置
- **GPU 未识别**：运行 doctor 检查，或通过 `--device` 手动指定（cuda/mps/xpu/cpu）
- **海外网络**：设置 `TTS_SKILL_FORCE_REGION=global` 关闭国内镜像
- **transcribe 命令不可用**：运行 `python -m tts_skill setup` 安装 ASR 依赖（faster-whisper + imageio-ffmpeg）
- **macOS 转写慢**：faster-whisper 不支持 Apple MPS，自动回退到 CPU + int8；可改用 `--model small` 加速
- **视频转写失败**：imageio-ffmpeg 自带跨平台 ffmpeg 二进制；如仍失败可手动安装系统 ffmpeg 并设置 `FFMPEG_BINARY` 环境变量
