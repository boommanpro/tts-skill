---
name: tts-skill
description: |
  OmniVoice 声音克隆与生成工具：输入文本/参考音频/声音属性，自动完成 600+ 语种的零样本文本转语音、声音克隆、声音设计、批量生成。
  三种入口：(1)Web UI 可视化调参 (2)CLI 一行命令生成 (3)批量生成+可复现命令输出。
  触发词：「文本转语音」「声音克隆」「语音合成」「生成语音」「TTS」「调参生成」「批量生成音频」「声音设计」「复现语音」。
  English triggers: "text to speech", "voice cloning", "speech synthesis", "generate audio", "TTS", "voice design", "batch generate audio".
---

# OmniVoice TTS Skill

> 「一行命令，让文本发出任何人的声音。」

基于开源项目 [OmniVoice](https://github.com/k2-fsa/OmniVoice) 的声音克隆与生成工具。支持 600+ 语种零样本文本转语音，跨平台（Linux/macOS/Windows）自动安装，国内镜像加速。

## 何时调用此 Skill

当用户有以下需求时调用：

- 文本转语音（TTS）/ 语音合成 / 生成语音
- 声音克隆（用参考音频复刻声音）
- 声音设计（按性别、年龄、音调、口音等属性生成）
- 批量生成音频
- 需要可视化调参后生成语音
- 需要可复现的语音生成（随机种子）

## 一键执行（无需思考，直接运行）

### 1. 启动 Web UI（可视化调参，推荐）

```bash
python -m tts_skill web
```

首次运行会自动安装环境（约 10-20 分钟，国内镜像加速）。安装完成后浏览器访问 `http://localhost:7860`。

Web UI 功能：

- **声音克隆**：上传参考音频 + 输入文本 → 克隆声音
- **声音设计**：选择性别/年龄/音调/口音/方言 → 生成声音
- **自动声音**：仅输入文本 → 模型自动选择声音
- **随机种子配置**：固定种子可复现，随机种子可探索
- **批量生成**：默认 5 个，每个不同种子
- **历史记录**：查看、重命名、删除、查看复现命令

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

### 3. 其他命令

```bash
python -m tts_skill setup          # 单独安装环境
python -m tts_skill setup --force  # 强制重新安装
python -m tts_skill history        # 查看历史记录
python -m tts_skill history --show_cmd  # 显示所有复现命令
python -m tts_skill doctor         # 诊断环境问题
python -m tts_skill --version      # 查看版本
```

## 执行规则（AI Agent 必读）

1. **无需思考，直接运行命令**：所有逻辑已封装在 Python 脚本中
2. **首次使用先 setup**：如果用户首次使用，先运行 `python -m tts_skill setup`（或直接运行 web/infer，会自动安装）
3. **国内环境自动加速**：脚本默认启用国内 pip 镜像和 HuggingFace 镜像（`hf-mirror.com`）
4. **跨平台自动适配**：自动检测 Linux/macOS/Windows 和 GPU 类型（CUDA/MPS/XPU/CPU）
5. **Web 优先**：用户需要调参时，启动 Web UI；用户明确要 CLI 时，用 infer 命令
6. **复现命令**：生成后务必展示给用户的可复现命令，方便后续复现

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
