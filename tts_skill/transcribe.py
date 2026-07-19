"""语音转文字（ASR）模块：基于 faster-whisper，支持音频和视频输入。

功能：
- 输入音频文件（wav/mp3/flac/m4a/aac/ogg/...）或视频文件（mp4/mov/mkv/avi/webm/...）
- 视频自动通过 ffmpeg 提取音轨
- 输出 JSON（含词级时间戳）+ 纯文本预览
- 支持语言自动检测或指定
- 支持转写（transcribe）和翻译成英文（translate）
- 跨平台（Linux/macOS/Windows），国内镜像加速

用法：
    python -m tts_skill transcribe --input video.mp4
    python -m tts_skill transcribe --input audio.mp3 --language zh
    python -m tts_skill transcribe --input video.mp4 --model large-v3 --task translate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from tts_skill.utils import (
    OUTPUT_DIR,
    ensure_output_dir,
    get_platform,
    set_hf_mirror_env,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 支持的模型大小（faster-whisper 标准）
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# 默认模型
DEFAULT_ASR_MODEL = "large-v3"

# 视频扩展名（需要先用 ffmpeg 提取音轨）
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp", ".ogv",
}

# 音频扩展名（faster-whisper 可直接读取，但内部仍会用 ffmpeg 转码）
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma",
    ".aiff", ".aif", ".opus", ".weba", ".oga",
}


# ---------------------------------------------------------------------------
# 历史记录（独立存储，避免污染 TTS 的 history.json）
# ---------------------------------------------------------------------------

# 转写历史记录文件
TRANSCRIBE_HISTORY_FILE = Path(os.environ.get(
    "TTS_SKILL_TRANSCRIBE_HISTORY",
    str(Path(__file__).resolve().parent.parent / "transcribe_history.json"),
))


@dataclass
class TranscribeRecord:
    """单次转写记录。"""

    id: str
    name: str
    timestamp: str
    input_file: str
    file_type: str  # "audio" | "video"
    language: Optional[str]  # 检测到的语言代码（如 "zh", "en"）
    language_probability: float
    duration: float  # 音频总时长（秒）
    model_size: str
    task: str  # "transcribe" | "translate"
    device: str
    text: str  # 完整纯文本
    segment_count: int
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscribeRecord":
        valid_keys = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TranscribeResult:
    """转写结果（在内存中传递，不持久化）。"""

    text: str  # 完整纯文本
    language: str  # 检测到的语言代码
    language_probability: float  # 语言检测置信度（0-1）
    duration: float  # 音频总时长（秒）
    duration_after_vad: Optional[float]  # VAD 后时长（秒）
    segments: list[dict[str, Any]]  # 每段：start, end, text, words, avg_logprob, no_speech_prob, ...
    model_size: str
    task: str
    device: str
    output_files: list[str]  # 保存的文件路径


# ---------------------------------------------------------------------------
# 文件类型检测
# ---------------------------------------------------------------------------


def detect_file_type(path: Path) -> str:
    """检测文件是音频还是视频。

    返回: "audio" | "video"

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件扩展名
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"不是文件: {path}")

    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    raise ValueError(
        f"不支持的文件扩展名: {ext}\n"
        f"支持的视频: {sorted(VIDEO_EXTENSIONS)}\n"
        f"支持的音频: {sorted(AUDIO_EXTENSIONS)}"
    )


# ---------------------------------------------------------------------------
# FFmpeg 二进制查找
# ---------------------------------------------------------------------------


def get_ffmpeg_binary() -> str:
    """获取 ffmpeg 二进制路径。

    优先使用 imageio-ffmpeg 提供的跨平台二进制（无需用户手动安装）。
    如果不可用，回退到系统 ffmpeg。
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception) as e:
        logger.warning("imageio-ffmpeg 不可用: %s，尝试系统 ffmpeg", e)
        # 回退：检查系统 PATH 中的 ffmpeg
        plat = get_platform()
        ffmpeg_name = "ffmpeg.exe" if plat == "windows" else "ffmpeg"
        try:
            result = subprocess.run(
                [ffmpeg_name, "-version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ffmpeg_name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise RuntimeError(
            "未找到 ffmpeg。请安装依赖：pip install imageio-ffmpeg\n"
            "或手动安装 ffmpeg 并加入 PATH。"
        )


def ensure_ffmpeg_env() -> None:
    """确保 faster-whisper 能找到 ffmpeg。

    faster-whisper 内部使用 ffmpeg 处理音频输入。设置 FFMPEG_BINARY 环境变量
    让它使用 imageio-ffmpeg 提供的二进制，无需用户手动安装 ffmpeg。
    """
    if os.environ.get("TTS_SKILL_SKIP_FFMPEG_ENV"):
        return
    try:
        ffmpeg_path = get_ffmpeg_binary()
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg_path)
        # 同时把 ffmpeg 所在目录加入 PATH（faster-whisper 部分代码会查 PATH）
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        path_sep = ";" if get_platform() == "windows" else ":"
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path.split(path_sep):
            os.environ["PATH"] = ffmpeg_dir + path_sep + current_path
    except Exception as e:
        logger.warning("无法设置 ffmpeg 环境: %s", e)


# ---------------------------------------------------------------------------
# 视频提取音轨
# ---------------------------------------------------------------------------


def extract_audio_from_video(
    video_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """从视频提取音轨为 16kHz 单声道 wav。

    Args:
        video_path: 视频文件路径
        output_path: 输出 wav 路径，None 则创建临时文件

    Returns:
        输出 wav 文件路径
    """
    ffmpeg = get_ffmpeg_binary()

    if output_path is None:
        tmp_dir = Path(tempfile.gettempdir())
        output_path = tmp_dir / f"tts_skill_{uuid.uuid4().hex[:8]}.wav"

    # 使用 16kHz 单声道 wav（faster-whisper 的标准输入格式）
    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vn",  # 不要视频
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",  # 16kHz 采样率
        "-ac", "1",  # 单声道
        "-y",  # 覆盖输出
        str(output_path),
    ]
    logger.info("提取音轨: %s -> %s", video_path.name, output_path.name)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 提取音轨失败 (code={result.returncode}):\n"
            f"{result.stderr[-2000:] if result.stderr else '无错误输出'}"
        )
    return output_path


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, Any] = {}


def _resolve_compute_type(device: str, compute_type: Optional[str]) -> str:
    """根据设备推断默认 compute_type。"""
    if compute_type:
        return compute_type
    # CPU: int8（最快且内存占用最低）
    # CUDA: float16（速度与精度平衡）
    # MPS: faster-whisper 不支持 MPS，回退到 CPU + int8
    if device == "cuda":
        return "float16"
    return "int8"


def _detect_asr_device() -> str:
    """检测 ASR 推理设备。

    faster-whisper 基于 CTranslate2，支持 cuda 和 cpu。
    注意：不支持 Apple MPS（CTranslate2 限制），所以 macOS 都走 cpu（int8 仍然很快）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    # 检查 NVIDIA GPU（torch 未安装时）
    try:
        from tts_skill.utils import has_nvidia_gpu

        if has_nvidia_gpu():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_transcriber(
    model_size: str = DEFAULT_ASR_MODEL,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    download_root: Optional[str] = None,
):
    """加载并缓存 faster-whisper 模型。

    Args:
        model_size: 模型大小（tiny/base/small/medium/large-v3）
        device: 设备（cuda/cpu），None 自动检测
        compute_type: 计算类型，None 自动推断
        download_root: 模型下载目录，None 使用默认（~/.cache/huggingface）

    Returns:
        faster_whisper.WhisperModel 实例
    """
    from faster_whisper import WhisperModel

    if device is None:
        device = _detect_asr_device()
    ct = _resolve_compute_type(device, compute_type)

    cache_key = f"{model_size}|{device}|{ct}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 设置 HF 镜像（国内加速模型下载）
    set_hf_mirror_env()

    logger.info(
        "加载 faster-whisper 模型 %s (device=%s, compute_type=%s)...",
        model_size, device, ct,
    )
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=ct,
            download_root=download_root,
        )
    except Exception as e:
        # 回退到 CPU + int8
        if device != "cpu":
            logger.warning("加载失败 (%s)，回退到 CPU + int8: %s", device, e)
            device = "cpu"
            ct = "int8"
            cache_key = f"{model_size}|{device}|{ct}"
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key]
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=ct,
                download_root=download_root,
            )
        else:
            raise

    _MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# 核心转写逻辑
# ---------------------------------------------------------------------------


def transcribe_file(
    input_path: str | Path,
    *,
    model_size: str = DEFAULT_ASR_MODEL,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,  # None=自动检测；"zh"/"en"/...
    task: str = "transcribe",  # "transcribe" | "translate"（翻译成英文）
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    output_dir: Optional[Path] = None,
    name: Optional[str] = None,
    keep_temp_audio: bool = False,
) -> TranscribeResult:
    """转写音频或视频文件。

    Args:
        input_path: 输入文件路径（音频或视频）
        model_size: 模型大小
        device: 设备（自动检测）
        compute_type: 计算类型（自动推断）
        language: 语言代码（None 自动检测）
        task: transcribe=转写原语言, translate=翻译成英文
        beam_size: beam search 大小
        word_timestamps: 是否生成词级时间戳
        vad_filter: 是否启用 VAD 过滤静音
        output_dir: 输出目录（默认 outputs/）
        name: 记录名称
        keep_temp_audio: 是否保留从视频提取的临时 wav

    Returns:
        TranscribeResult
    """
    input_path = Path(input_path).resolve()
    file_type = detect_file_type(input_path)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if task not in ("transcribe", "translate"):
        raise ValueError(f"task 必须是 'transcribe' 或 'translate'，得到: {task}")

    # 确保 ffmpeg 可用
    ensure_ffmpeg_env()

    # 视频先提取音轨
    temp_audio_path: Optional[Path] = None
    audio_input: Path = input_path
    if file_type == "video":
        temp_audio_path = extract_audio_from_video(input_path)
        audio_input = temp_audio_path

    try:
        # 加载模型
        if device is None:
            device = _detect_asr_device()
        model = get_transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

        # 转写
        logger.info("开始转写 (%s, task=%s, language=%s)...",
                    audio_input.name, task, language or "auto")
        segments_iter, info = model.transcribe(
            str(audio_input),
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        # 收集 segments（faster-whisper 返回的是生成器）
        segments: list[dict[str, Any]] = []
        full_text_parts: list[str] = []
        for seg in segments_iter:
            seg_dict: dict[str, Any] = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
                "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
                "compression_ratio": float(seg.compression_ratio) if seg.compression_ratio is not None else None,
            }
            if word_timestamps and seg.words:
                seg_dict["words"] = [
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "word": w.word,
                        "probability": float(w.probability) if w.probability is not None else None,
                    }
                    for w in seg.words
                ]
            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts).strip()

        # 构造结果
        result = TranscribeResult(
            text=full_text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration=float(info.duration),
            duration_after_vad=float(info.duration_after_vad) if info.duration_after_vad else None,
            segments=segments,
            model_size=model_size,
            task=task,
            device=device,
            output_files=[],
        )

        # 保存输出文件
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_name = input_path.stem
        record_id = str(uuid.uuid4())[:8]

        # JSON 输出（包含完整元数据 + 词级时间戳）
        json_filename = f"{base_name}_{timestamp_str}_{record_id}.json"
        json_path = output_dir / json_filename
        json_data = {
            "id": record_id,
            "input_file": str(input_path),
            "file_type": file_type,
            "text": full_text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "duration_after_vad": result.duration_after_vad,
            "segments": segments,
            "model_size": model_size,
            "task": task,
            "device": device,
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result.output_files.append(str(json_path))
        logger.info("已保存 JSON: %s", json_path)

        # 同时保存纯文本（方便直接使用）
        txt_filename = f"{base_name}_{timestamp_str}_{record_id}.txt"
        txt_path = output_dir / txt_filename
        txt_path.write_text(full_text, encoding="utf-8")
        result.output_files.append(str(txt_path))
        logger.info("已保存文本: %s", txt_path)

        # 创建历史记录
        record = TranscribeRecord(
            id=record_id,
            name=name or f"{base_name}_{task}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            input_file=str(input_path),
            file_type=file_type,
            language=result.language,
            language_probability=result.language_probability,
            duration=result.duration,
            model_size=model_size,
            task=task,
            device=device,
            text=full_text,
            segment_count=len(segments),
            output_files=result.output_files,
        )
        _add_transcribe_record(record)

        return result

    finally:
        # 清理临时音轨
        if temp_audio_path and not keep_temp_audio:
            try:
                temp_audio_path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 转写历史记录管理
# ---------------------------------------------------------------------------


def _load_transcribe_history_raw() -> list[dict[str, Any]]:
    """加载原始转写历史记录。"""
    if not TRANSCRIBE_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(TRANSCRIBE_HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def load_transcribe_history() -> list[TranscribeRecord]:
    """加载所有转写历史记录（按时间倒序）。"""
    raw = _load_transcribe_history_raw()
    records = [TranscribeRecord.from_dict(item) for item in raw]
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records


def _save_transcribe_history(records: list[TranscribeRecord]) -> None:
    """保存转写历史记录（按时间正序存储）。"""
    sorted_records = sorted(records, key=lambda r: r.timestamp)
    data = [r.to_dict() for r in sorted_records]
    TRANSCRIBE_HISTORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _add_transcribe_record(record: TranscribeRecord) -> None:
    """添加一条转写记录到历史记录。"""
    records = load_transcribe_history()
    records.append(record)
    _save_transcribe_history(records)


def get_transcribe_record(record_id: str) -> Optional[TranscribeRecord]:
    """根据 ID 获取转写记录。"""
    for r in load_transcribe_history():
        if r.id == record_id:
            return r
    return None


def update_transcribe_record_name(record_id: str, new_name: str) -> bool:
    """更新转写记录名称，返回是否成功。"""
    records = load_transcribe_history()
    for r in records:
        if r.id == record_id:
            r.name = new_name
            _save_transcribe_history(records)
            return True
    return False


def delete_transcribe_record(record_id: str) -> bool:
    """删除一条转写记录（同时删除输出文件），返回是否成功。"""
    records = load_transcribe_history()
    remaining = []
    deleted = False
    for r in records:
        if r.id == record_id:
            deleted = True
            for f in r.output_files:
                try:
                    Path(f).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            remaining.append(r)
    if deleted:
        _save_transcribe_history(remaining)
    return deleted


# ---------------------------------------------------------------------------
# 复现命令生成
# ---------------------------------------------------------------------------


def _shell_quote(s: str) -> str:
    """对字符串进行 shell 转义（跨平台兼容）。

    只允许 ASCII 字母数字和 `._-/` 不转义，其他（含中文、空格、特殊符号）
    一律用双引号包裹。
    """
    if not s:
        return "''"
    if all(c.isascii() and (c.isalnum() or c in "._-/") for c in s):
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_transcribe_reproducible_command(record: TranscribeRecord) -> str:
    """根据转写记录生成可复现的 CLI 命令字符串。"""
    from tts_skill.utils import get_python_executable

    parts = [get_python_executable(), "-m", "tts_skill", "transcribe"]
    parts += ["--input", _shell_quote(record.input_file)]
    parts += ["--model", record.model_size]
    parts += ["--task", record.task]
    if record.language:
        parts += ["--language", record.language]
    parts += ["--device", record.device]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------


def build_transcribe_parser() -> argparse.ArgumentParser:
    """构建 transcribe 命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="tts_skill transcribe",
        description="语音转文字：将音频或视频文件转写为文字（基于 faster-whisper）",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 必填
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径（音频或视频）")

    # 模型与设备
    parser.add_argument("--model", type=str, default=DEFAULT_ASR_MODEL,
                        choices=SUPPORTED_MODELS,
                        help=f"模型大小（默认 {DEFAULT_ASR_MODEL}）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（cuda/cpu，自动检测）")
    parser.add_argument("--compute_type", type=str, default=None,
                        help="计算类型（如 int8, float16, float32），自动推断")

    # 转写选项
    parser.add_argument("--language", type=str, default=None,
                        help="语言代码（如 zh, en, ja），留空自动检测")
    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="transcribe=转写原语言, translate=翻译成英文（默认 transcribe）")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="beam search 大小（默认 5）")
    parser.add_argument("--word_timestamps", type=lambda v: v.lower() in ("true", "1", "yes"),
                        default=True, help="生成词级时间戳（默认 true）")
    parser.add_argument("--vad_filter", type=lambda v: v.lower() in ("true", "1", "yes"),
                        default=True, help="VAD 过滤静音（默认 true）")

    # 输出
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--name", type=str, default=None,
                        help="记录名称（用于历史记录）")

    return parser


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def run_transcribe(args: argparse.Namespace) -> int:
    """执行转写流程，返回退出码。"""
    set_hf_mirror_env()
    ensure_output_dir()

    # 检查 ASR 依赖
    try:
        import faster_whisper  # noqa: F401

        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        logger.error(
            "语音转文字依赖未安装，请运行: python -m tts_skill setup\n"
            "或单独安装: pip install faster-whisper imageio-ffmpeg"
        )
        return 1

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("输入文件不存在: %s", input_path)
        return 1

    try:
        file_type = detect_file_type(input_path)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info("输入: %s (%s)", input_path, file_type)

    try:
        result = transcribe_file(
            input_path,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            word_timestamps=args.word_timestamps,
            vad_filter=args.vad_filter,
            output_dir=Path(args.output_dir),
            name=args.name,
        )
    except Exception as e:
        logger.error("转写失败: %s", e, exc_info=True)
        return 1

    # 输出结果
    print("\n" + "=" * 60)
    print("转写完成！")
    print("=" * 60)
    print(f"输入文件: {input_path.name} ({file_type})")
    print(f"检测语言: {result.language} (置信度 {result.language_probability:.2%})")
    print(f"音频时长: {result.duration:.1f}s"
          + (f" (VAD 后 {result.duration_after_vad:.1f}s)" if result.duration_after_vad else ""))
    print(f"分段数量: {len(result.segments)}")
    print(f"模型: {result.model_size} (device={result.device}, task={result.task})")
    print(f"输出文件:")
    for f in result.output_files:
        print(f"  - {f}")

    # 可复现命令（从最新历史记录中取，确保参数完整）
    records = load_transcribe_history()
    if records:
        latest = records[0]
        reproducible_cmd = build_transcribe_reproducible_command(latest)
        print()
        print("可复现命令（复制以下命令可重新转写相同结果）:")
        print("-" * 60)
        print(reproducible_cmd)
        print("-" * 60)

    # 打印转写文本预览（前 500 字）
    print("\n转写文本预览（前 500 字）:")
    print("-" * 60)
    preview = result.text[:500]
    if len(result.text) > 500:
        preview += "..."
    print(preview)
    print("-" * 60)

    return 0
