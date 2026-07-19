"""测试 tts_skill.transcribe 语音转文字模块。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from tts_skill import transcribe
from tts_skill.transcribe import (
    AUDIO_EXTENSIONS,
    DEFAULT_ASR_MODEL,
    SUPPORTED_MODELS,
    TRANSCRIBE_HISTORY_FILE,
    TranscribeRecord,
    TranscribeResult,
    _resolve_compute_type,
    _shell_quote,
    build_transcribe_parser,
    build_transcribe_reproducible_command,
    delete_transcribe_record,
    detect_file_type,
    extract_audio_from_video,
    get_transcriber,
    load_transcribe_history,
    transcribe_file,
    update_transcribe_record_name,
)


# ---------------------------------------------------------------------------
# 文件类型检测
# ---------------------------------------------------------------------------


class TestDetectFileType:
    """测试文件类型检测。"""

    def test_detect_audio_wav(self, tmp_path: Path):
        """检测 wav 音频。"""
        f = tmp_path / "test.wav"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "audio"

    def test_detect_audio_mp3(self, tmp_path: Path):
        """检测 mp3 音频。"""
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "audio"

    def test_detect_audio_flac(self, tmp_path: Path):
        """检测 flac 音频。"""
        f = tmp_path / "audio.flac"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "audio"

    def test_detect_video_mp4(self, tmp_path: Path):
        """检测 mp4 视频。"""
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "video"

    def test_detect_video_mov(self, tmp_path: Path):
        """检测 mov 视频。"""
        f = tmp_path / "clip.mov"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "video"

    def test_detect_video_mkv(self, tmp_path: Path):
        """检测 mkv 视频。"""
        f = tmp_path / "video.mkv"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "video"

    def test_detect_case_insensitive(self, tmp_path: Path):
        """扩展名大小写不敏感。"""
        f = tmp_path / "TEST.MP4"
        f.write_bytes(b"fake")
        assert detect_file_type(f) == "video"

        f2 = tmp_path / "Audio.WAV"
        f2.write_bytes(b"fake")
        assert detect_file_type(f2) == "audio"

    def test_detect_not_found(self, tmp_path: Path):
        """文件不存在抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            detect_file_type(tmp_path / "nonexistent.wav")

    def test_detect_unsupported_extension(self, tmp_path: Path):
        """不支持的扩展名抛 ValueError。"""
        f = tmp_path / "file.xyz"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="不支持的文件扩展名"):
            detect_file_type(f)

    def test_audio_extensions_complete(self):
        """音频扩展名集合包含常见格式。"""
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]:
            assert ext in AUDIO_EXTENSIONS

    def test_video_extensions_complete(self):
        """视频扩展名集合包含常见格式。"""
        for ext in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
            assert ext in transcribe.VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# compute_type 推断
# ---------------------------------------------------------------------------


class TestResolveComputeType:
    """测试 compute_type 推断。"""

    def test_cuda_default_float16(self):
        """CUDA 默认 float16。"""
        assert _resolve_compute_type("cuda", None) == "float16"

    def test_cpu_default_int8(self):
        """CPU 默认 int8。"""
        assert _resolve_compute_type("cpu", None) == "int8"

    def test_mps_fallback_to_cpu(self):
        """faster-whisper 不支持 MPS，回退 int8。"""
        assert _resolve_compute_type("cpu", None) == "int8"

    def test_explicit_compute_type(self):
        """显式指定时覆盖默认。"""
        assert _resolve_compute_type("cuda", "int8") == "int8"
        assert _resolve_compute_type("cpu", "float32") == "float32"


# ---------------------------------------------------------------------------
# shell 转义
# ---------------------------------------------------------------------------


class TestShellQuote:
    """测试 shell 转义。"""

    def test_simple_alnum(self):
        """纯字母数字不转义。"""
        assert _shell_quote("hello123") == "hello123"

    def test_path(self):
        """路径不转义。"""
        assert _shell_quote("/tmp/file.wav") == "/tmp/file.wav"

    def test_empty(self):
        """空字符串。"""
        assert _shell_quote("") == "''"

    def test_with_spaces(self):
        """含空格用双引号包裹。"""
        assert _shell_quote("hello world") == '"hello world"'

    def test_with_chinese(self):
        """中文字符用双引号包裹。"""
        result = _shell_quote("你好世界")
        assert result.startswith('"') and result.endswith('"')
        assert "你好世界" in result

    def test_with_double_quote(self):
        """含双引号转义。"""
        result = _shell_quote('say "hi"')
        assert '\\"' in result


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class TestTranscribeRecord:
    """测试 TranscribeRecord 数据类。"""

    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict 往返一致。"""
        record = TranscribeRecord(
            id="abc123",
            name="test",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.95,
            duration=120.5,
            model_size="large-v3",
            task="transcribe",
            device="cuda",
            text="你好世界",
            segment_count=3,
            output_files=["/tmp/out.json", "/tmp/out.txt"],
        )
        d = record.to_dict()
        assert d["id"] == "abc123"
        assert d["text"] == "你好世界"
        assert d["output_files"] == ["/tmp/out.json", "/tmp/out.txt"]

        # 反序列化
        r2 = TranscribeRecord.from_dict(d)
        assert r2.id == record.id
        assert r2.text == record.text
        assert r2.output_files == record.output_files

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict 忽略未知字段。"""
        record = TranscribeRecord.from_dict({
            "id": "x",
            "name": "y",
            "timestamp": "2026-07-19 10:00:00",
            "input_file": "/tmp/x.wav",
            "file_type": "audio",
            "language": "en",
            "language_probability": 0.9,
            "duration": 10.0,
            "model_size": "base",
            "task": "transcribe",
            "device": "cpu",
            "text": "hi",
            "segment_count": 1,
            "output_files": [],
            "unknown_key": "ignored",
        })
        assert record.id == "x"
        assert not hasattr(record, "unknown_key")

    def test_from_dict_handles_missing_optional(self):
        """from_dict 处理缺失的可选字段。"""
        record = TranscribeRecord.from_dict({
            "id": "x",
            "name": "y",
            "timestamp": "2026-07-19 10:00:00",
            "input_file": "/tmp/x.wav",
            "file_type": "audio",
            "language": None,
            "language_probability": 0.0,
            "duration": 10.0,
            "model_size": "base",
            "task": "transcribe",
            "device": "cpu",
            "text": "hi",
            "segment_count": 1,
            # output_files 缺失，应使用默认值
        })
        assert record.output_files == []


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


class TestTranscribeParser:
    """测试 transcribe 命令参数解析器。"""

    def test_parser_basic_args(self):
        """基本参数。"""
        parser = build_transcribe_parser()
        args = parser.parse_args(["--input", "test.mp4"])
        assert args.input == "test.mp4"
        assert args.model == DEFAULT_ASR_MODEL
        assert args.task == "transcribe"
        assert args.beam_size == 5
        assert args.word_timestamps is True
        assert args.vad_filter is True

    def test_parser_required_input(self):
        """--input 必填。"""
        parser = build_transcribe_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_model_choices(self):
        """model 受 SUPPORTED_MODELS 限制。"""
        parser = build_transcribe_parser()
        for m in SUPPORTED_MODELS:
            args = parser.parse_args(["--input", "x.wav", "--model", m])
            assert args.model == m

        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "x.wav", "--model", "invalid"])

    def test_parser_task_choices(self):
        """task 受 transcribe/translate 限制。"""
        parser = build_transcribe_parser()
        args = parser.parse_args(["--input", "x.wav", "--task", "translate"])
        assert args.task == "translate"

        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "x.wav", "--task", "invalid"])

    def test_parser_language(self):
        """language 可选。"""
        parser = build_transcribe_parser()
        args = parser.parse_args(["--input", "x.wav", "--language", "zh"])
        assert args.language == "zh"

    def test_parser_bool_args(self):
        """bool 参数解析。"""
        parser = build_transcribe_parser()
        args = parser.parse_args([
            "--input", "x.wav",
            "--word_timestamps", "false",
            "--vad_filter", "0",
        ])
        assert args.word_timestamps is False
        assert args.vad_filter is False

    def test_parser_beam_size(self):
        """beam_size 整数。"""
        parser = build_transcribe_parser()
        args = parser.parse_args(["--input", "x.wav", "--beam_size", "10"])
        assert args.beam_size == 10


# ---------------------------------------------------------------------------
# 复现命令生成
# ---------------------------------------------------------------------------


class TestReproducibleCommand:
    """测试可复现命令生成。"""

    def test_basic_command(self):
        """基本命令格式。"""
        record = TranscribeRecord(
            id="abc",
            name="test",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.95,
            duration=120.0,
            model_size="large-v3",
            task="transcribe",
            device="cuda",
            text="你好",
            segment_count=1,
            output_files=[],
        )
        cmd = build_transcribe_reproducible_command(record)
        assert "transcribe" in cmd
        assert "/tmp/test.mp4" in cmd
        assert "large-v3" in cmd
        assert "transcribe" in cmd
        assert "zh" in cmd
        assert "cuda" in cmd

    def test_command_with_chinese_path(self):
        """含中文路径的转义。"""
        record = TranscribeRecord(
            id="abc",
            name="test",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/中文路径/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.95,
            duration=60.0,
            model_size="large-v3",
            task="transcribe",
            device="cpu",
            text="你好",
            segment_count=1,
            output_files=[],
        )
        cmd = build_transcribe_reproducible_command(record)
        # 中文路径应该被引号包裹
        assert '"' in cmd


# ---------------------------------------------------------------------------
# 历史记录管理
# ---------------------------------------------------------------------------


class TestTranscribeHistory:
    """测试转写历史记录管理。"""

    def test_load_empty_history(self, tmp_path: Path, monkeypatch):
        """空历史。"""
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", tmp_path / "asr_history.json")
        assert load_transcribe_history() == []

    def test_add_and_load(self, tmp_path: Path, monkeypatch):
        """添加并加载。"""
        history_file = tmp_path / "asr_history.json"
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", history_file)

        record = TranscribeRecord(
            id="test1",
            name="test",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.95,
            duration=120.0,
            model_size="large-v3",
            task="transcribe",
            device="cuda",
            text="你好",
            segment_count=1,
            output_files=["/tmp/out.json"],
        )
        transcribe._add_transcribe_record(record)

        loaded = load_transcribe_history()
        assert len(loaded) == 1
        assert loaded[0].id == "test1"
        assert loaded[0].text == "你好"

    def test_load_sorted_by_time_desc(self, tmp_path: Path, monkeypatch):
        """按时间倒序。"""
        history_file = tmp_path / "asr_history.json"
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", history_file)

        for i, ts in enumerate(["2026-07-19 10:00:00", "2026-07-20 10:00:00", "2026-07-19 12:00:00"]):
            transcribe._add_transcribe_record(TranscribeRecord(
                id=f"r{i}",
                name=f"record{i}",
                timestamp=ts,
                input_file=f"/tmp/{i}.mp4",
                file_type="video",
                language="zh",
                language_probability=0.9,
                duration=10.0,
                model_size="base",
                task="transcribe",
                device="cpu",
                text="x",
                segment_count=1,
                output_files=[],
            ))

        loaded = load_transcribe_history()
        # 最新的在前
        assert loaded[0].id == "r1"  # 2026-07-20
        assert loaded[1].id == "r2"  # 2026-07-19 12:00
        assert loaded[2].id == "r0"  # 2026-07-19 10:00

    def test_update_name(self, tmp_path: Path, monkeypatch):
        """更新名称。"""
        history_file = tmp_path / "asr_history.json"
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", history_file)

        transcribe._add_transcribe_record(TranscribeRecord(
            id="abc",
            name="old",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.9,
            duration=10.0,
            model_size="base",
            task="transcribe",
            device="cpu",
            text="x",
            segment_count=1,
            output_files=[],
        ))

        assert update_transcribe_record_name("abc", "new name") is True
        loaded = load_transcribe_history()
        assert loaded[0].name == "new name"

        assert update_transcribe_record_name("nonexistent", "x") is False

    def test_delete_record(self, tmp_path: Path, monkeypatch):
        """删除记录。"""
        history_file = tmp_path / "asr_history.json"
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", history_file)

        # 创建临时输出文件
        out_file = tmp_path / "out.json"
        out_file.write_text("{}")

        transcribe._add_transcribe_record(TranscribeRecord(
            id="del1",
            name="to delete",
            timestamp="2026-07-19 10:00:00",
            input_file="/tmp/test.mp4",
            file_type="video",
            language="zh",
            language_probability=0.9,
            duration=10.0,
            model_size="base",
            task="transcribe",
            device="cpu",
            text="x",
            segment_count=1,
            output_files=[str(out_file)],
        ))

        assert delete_transcribe_record("del1") is True
        assert not out_file.exists()  # 输出文件应被删除
        assert load_transcribe_history() == []

        # 删除不存在的记录
        assert delete_transcribe_record("nonexistent") is False

    def test_load_corrupt_json(self, tmp_path: Path, monkeypatch):
        """损坏的 JSON 文件返回空列表。"""
        history_file = tmp_path / "asr_history.json"
        history_file.write_text("not a valid json {{{")
        monkeypatch.setattr(transcribe, "TRANSCRIBE_HISTORY_FILE", history_file)

        assert load_transcribe_history() == []


# ---------------------------------------------------------------------------
# ffmpeg 相关
# ---------------------------------------------------------------------------


class TestFFmpeg:
    """测试 ffmpeg 二进制查找和视频音轨提取。"""

    def test_get_ffmpeg_binary_with_imageio(self, monkeypatch):
        """使用 imageio-ffmpeg 提供的二进制。"""
        fake_path = "/fake/ffmpeg"
        mock_module = mock.MagicMock()
        mock_module.get_ffmpeg_exe.return_value = fake_path

        # 模拟 import imageio_ffmpeg 成功
        monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", mock_module)
        result = transcribe.get_ffmpeg_binary()
        assert result == fake_path

    def test_get_ffmpeg_binary_fallback_to_system(self, monkeypatch):
        """imageio-ffmpeg 不可用时回退到系统 ffmpeg。"""
        # 模拟 imageio_ffmpeg 导入失败
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # 模拟系统 ffmpeg 可用
        mock_run = mock.MagicMock()
        mock_run.return_value = mock.MagicMock(returncode=0)
        monkeypatch.setattr(transcribe.subprocess, "run", mock_run)

        result = transcribe.get_ffmpeg_binary()
        # Windows 返回 ffmpeg.exe，其他返回 ffmpeg
        assert "ffmpeg" in result

    def test_get_ffmpeg_binary_not_found(self, monkeypatch):
        """找不到 ffmpeg 抛 RuntimeError。"""
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # 模拟系统 ffmpeg 不可用
        mock_run = mock.MagicMock(side_effect=FileNotFoundError())
        monkeypatch.setattr(transcribe.subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="未找到 ffmpeg"):
            transcribe.get_ffmpeg_binary()

    def test_ensure_ffmpeg_env_sets_env(self, monkeypatch):
        """ensure_ffmpeg_env 设置 FFMPEG_BINARY 环境变量。"""
        monkeypatch.delenv("TTS_SKILL_SKIP_FFMPEG_ENV", raising=False)
        monkeypatch.delenv("FFMPEG_BINARY", raising=False)
        monkeypatch.setattr(transcribe, "get_ffmpeg_binary", lambda: "/fake/ffmpeg")

        transcribe.ensure_ffmpeg_env()
        assert os.environ.get("FFMPEG_BINARY") == "/fake/ffmpeg"

    def test_ensure_ffmpeg_env_skip(self, monkeypatch):
        """TTS_SKILL_SKIP_FFMPEG_ENV 跳过设置。"""
        monkeypatch.setenv("TTS_SKILL_SKIP_FFMPEG_ENV", "1")
        monkeypatch.delenv("FFMPEG_BINARY", raising=False)
        monkeypatch.setattr(transcribe, "get_ffmpeg_binary", lambda: "/fake/ffmpeg")

        transcribe.ensure_ffmpeg_env()
        assert os.environ.get("FFMPEG_BINARY") is None

    def test_extract_audio_invokes_ffmpeg(self, tmp_path: Path, monkeypatch):
        """提取音轨调用 ffmpeg。"""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        output = tmp_path / "out.wav"

        # mock get_ffmpeg_binary
        monkeypatch.setattr(transcribe, "get_ffmpeg_binary", lambda: "/fake/ffmpeg")

        # mock subprocess.run（返回 returncode=0 表示成功）
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = ""
        completed.stderr = ""
        mock_run = mock.MagicMock(return_value=completed)
        monkeypatch.setattr(transcribe.subprocess, "run", mock_run)

        result = extract_audio_from_video(video, output)
        assert result == output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "/fake/ffmpeg" in cmd
        assert "-vn" in cmd
        assert "16000" in cmd  # 采样率
        assert "pcm_s16le" in cmd  # 16-bit PCM

    def test_extract_audio_failure_raises(self, tmp_path: Path, monkeypatch):
        """ffmpeg 失败时抛 RuntimeError。"""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        output = tmp_path / "out.wav"

        monkeypatch.setattr(transcribe, "get_ffmpeg_binary", lambda: "/fake/ffmpeg")
        completed = mock.MagicMock()
        completed.returncode = 1
        completed.stdout = ""
        completed.stderr = "some error"
        mock_run = mock.MagicMock(return_value=completed)
        monkeypatch.setattr(transcribe.subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="ffmpeg 提取音轨失败"):
            extract_audio_from_video(video, output)

    def test_extract_audio_default_output(self, tmp_path: Path, monkeypatch):
        """不指定输出路径时创建临时文件。"""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        monkeypatch.setattr(transcribe, "get_ffmpeg_binary", lambda: "/fake/ffmpeg")
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = ""
        completed.stderr = ""
        mock_run = mock.MagicMock(return_value=completed)
        monkeypatch.setattr(transcribe.subprocess, "run", mock_run)

        result = extract_audio_from_video(video)
        assert result.suffix == ".wav"
        assert "tts_skill_" in result.name


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


class TestModelLoading:
    """测试模型加载逻辑。"""

    def test_get_transcriber_caches(self, monkeypatch):
        """相同参数的模型被缓存。"""
        # mock WhisperModel
        mock_model_class = mock.MagicMock()
        mock_instance = mock.MagicMock()
        mock_model_class.return_value = mock_instance

        # 清空缓存
        transcribe._MODEL_CACHE.clear()

        # mock faster_whisper 模块
        mock_module = mock.MagicMock()
        mock_module.WhisperModel = mock_model_class
        monkeypatch.setitem(__import__("sys").modules, "faster_whisper", mock_module)

        # mock set_hf_mirror_env
        monkeypatch.setattr(transcribe, "set_hf_mirror_env", lambda: None)

        # 第一次调用：创建新实例
        m1 = get_transcriber(model_size="base", device="cpu", compute_type="int8")
        assert mock_model_class.call_count == 1
        assert m1 is mock_instance

        # 第二次相同参数：从缓存返回
        m2 = get_transcriber(model_size="base", device="cpu", compute_type="int8")
        assert mock_model_class.call_count == 1  # 没有再次创建
        assert m2 is mock_instance

    def test_get_transcriber_cuda_uses_float16(self, monkeypatch):
        """CUDA 设备默认 float16。"""
        mock_model_class = mock.MagicMock()
        mock_module = mock.MagicMock()
        mock_module.WhisperModel = mock_model_class
        monkeypatch.setitem(__import__("sys").modules, "faster_whisper", mock_module)
        monkeypatch.setattr(transcribe, "set_hf_mirror_env", lambda: None)
        transcribe._MODEL_CACHE.clear()

        get_transcriber(model_size="large-v3", device="cuda")
        # 检查传给 WhisperModel 的 compute_type
        _, kwargs = mock_model_class.call_args
        assert kwargs["compute_type"] == "float16"
        assert kwargs["device"] == "cuda"

    def test_get_transcriber_fallback_to_cpu(self, monkeypatch):
        """CUDA 加载失败时回退到 CPU。"""
        call_count = [0]

        def mock_init(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一次（cuda）抛异常
                raise RuntimeError("CUDA not available")
            # 第二次（cpu）成功
            return mock.MagicMock()

        mock_model_class = mock.MagicMock(side_effect=mock_init)
        mock_module = mock.MagicMock()
        mock_module.WhisperModel = mock_model_class
        monkeypatch.setitem(__import__("sys").modules, "faster_whisper", mock_module)
        monkeypatch.setattr(transcribe, "set_hf_mirror_env", lambda: None)
        transcribe._MODEL_CACHE.clear()

        # 应该回退成功
        m = get_transcriber(model_size="base", device="cuda")
        assert m is not None
        assert call_count[0] == 2  # 调用了 2 次


# ---------------------------------------------------------------------------
# 核心转写流程
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    """测试 transcribe_file 主流程（用 mock 模拟模型）。"""

    def _make_mock_segment(self, seg_id: int, start: float, end: float, text: str):
        """构造 mock segment。"""
        seg = mock.MagicMock()
        seg.id = seg_id
        seg.start = start
        seg.end = end
        seg.text = text
        seg.avg_logprob = -0.5
        seg.no_speech_prob = 0.01
        seg.compression_ratio = 1.5
        seg.words = None
        return seg

    def _make_mock_info(self, language="zh", probability=0.95, duration=10.0, vad_duration=None):
        """构造 mock info。"""
        info = mock.MagicMock()
        info.language = language
        info.language_probability = probability
        info.duration = duration
        info.duration_after_vad = vad_duration
        return info

    def _make_mock_model(self, segments, info):
        """构造 mock faster-whisper 模型。"""
        model = mock.MagicMock()
        model.transcribe.return_value = (iter(segments), info)
        return model

    def test_transcribe_audio_file(self, tmp_path: Path, monkeypatch):
        """转写音频文件。"""
        # 准备输入
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        # 输出目录
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        # mock 模型
        segments = [self._make_mock_segment(0, 0.0, 2.5, "你好"), self._make_mock_segment(1, 2.5, 5.0, "世界")]
        info = self._make_mock_info(language="zh", probability=0.98, duration=5.0)
        mock_model = self._make_mock_model(segments, info)

        # mock 依赖
        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        # 执行
        result = transcribe_file(audio_file, output_dir=out_dir)

        # 验证结果
        assert result.text == "你好 世界"
        assert result.language == "zh"
        assert result.language_probability == 0.98
        assert result.duration == 5.0
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "你好"
        assert result.segments[1]["text"] == "世界"
        assert len(result.output_files) == 2  # JSON + TXT

        # 验证文件已创建
        for f in result.output_files:
            assert Path(f).exists()

        # 验证 JSON 内容
        json_path = next(f for f in result.output_files if f.endswith(".json"))
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert data["text"] == "你好 世界"
        assert data["language"] == "zh"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][0]["end"] == 2.5

        # 验证 TXT 内容
        txt_path = next(f for f in result.output_files if f.endswith(".txt"))
        assert Path(txt_path).read_text(encoding="utf-8") == "你好 世界"

    def test_transcribe_video_file_extracts_audio(self, tmp_path: Path, monkeypatch):
        """转写视频文件时自动提取音轨。"""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        # mock 提取音轨（创建临时 wav）
        extracted = {"called": False}

        def fake_extract(video_path, output_path=None):
            extracted["called"] = True
            out = output_path or (tmp_path / "extracted.wav")
            out.write_bytes(b"fake wav")
            return out

        monkeypatch.setattr(transcribe, "extract_audio_from_video", fake_extract)

        # mock 模型
        segments = [self._make_mock_segment(0, 0.0, 1.0, "hello")]
        info = self._make_mock_info(language="en", probability=0.9, duration=1.0)
        mock_model = self._make_mock_model(segments, info)

        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        result = transcribe_file(video_file, output_dir=out_dir)

        # 验证调用了提取音轨
        assert extracted["called"] is True
        assert result.text == "hello"
        assert result.language == "en"

    def test_transcribe_with_word_timestamps(self, tmp_path: Path, monkeypatch):
        """词级时间戳。"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        # 构造带 words 的 segment
        seg = self._make_mock_segment(0, 0.0, 2.0, "hello world")
        word1 = mock.MagicMock()
        word1.start = 0.0
        word1.end = 1.0
        word1.word = "hello"
        word1.probability = 0.95
        word2 = mock.MagicMock()
        word2.start = 1.0
        word2.end = 2.0
        word2.word = "world"
        word2.probability = 0.92
        seg.words = [word1, word2]

        info = self._make_mock_info(language="en", duration=2.0)
        mock_model = self._make_mock_model([seg], info)

        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        result = transcribe_file(audio_file, output_dir=out_dir, word_timestamps=True)

        assert len(result.segments) == 1
        assert "words" in result.segments[0]
        assert len(result.segments[0]["words"]) == 2
        assert result.segments[0]["words"][0]["word"] == "hello"
        assert result.segments[0]["words"][0]["probability"] == 0.95

    def test_transcribe_invalid_task(self, tmp_path: Path):
        """无效 task 抛 ValueError。"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake")

        with pytest.raises(ValueError, match="task 必须是"):
            transcribe_file(audio_file, task="invalid_task")

    def test_transcribe_unsupported_file(self, tmp_path: Path):
        """不支持的文件类型抛 ValueError。"""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_bytes(b"fake")

        with pytest.raises(ValueError, match="不支持的文件扩展名"):
            transcribe_file(bad_file)

    def test_transcribe_translate_task(self, tmp_path: Path, monkeypatch):
        """translate 任务。"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        segments = [self._make_mock_segment(0, 0.0, 2.0, "Hello world")]
        info = self._make_mock_info(language="zh", duration=2.0)
        mock_model = self._make_mock_model(segments, info)

        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        result = transcribe_file(audio_file, output_dir=out_dir, task="translate")
        assert result.task == "translate"
        assert result.text == "Hello world"

        # 验证 model.transcribe 调用时 task="translate"
        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["task"] == "translate"

    def test_transcribe_cleans_temp_audio(self, tmp_path: Path, monkeypatch):
        """转写完成后清理临时音轨。"""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        temp_wav = tmp_path / "temp_extracted.wav"

        def fake_extract(video_path, output_path=None):
            temp_wav.write_bytes(b"fake wav")
            return temp_wav

        monkeypatch.setattr(transcribe, "extract_audio_from_video", fake_extract)
        segments = [self._make_mock_segment(0, 0.0, 1.0, "x")]
        info = self._make_mock_info(language="en", duration=1.0)
        mock_model = self._make_mock_model(segments, info)
        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        transcribe_file(video_file, output_dir=out_dir)
        # 默认 keep_temp_audio=False，临时文件应被删除
        assert not temp_wav.exists()

    def test_transcribe_keep_temp_audio(self, tmp_path: Path, monkeypatch):
        """keep_temp_audio=True 保留临时音轨。"""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        temp_wav = tmp_path / "temp_extracted.wav"

        def fake_extract(video_path, output_path=None):
            temp_wav.write_bytes(b"fake wav")
            return temp_wav

        monkeypatch.setattr(transcribe, "extract_audio_from_video", fake_extract)
        segments = [self._make_mock_segment(0, 0.0, 1.0, "x")]
        info = self._make_mock_info(language="en", duration=1.0)
        mock_model = self._make_mock_model(segments, info)
        monkeypatch.setattr(transcribe, "get_transcriber", lambda **kw: mock_model)
        monkeypatch.setattr(transcribe, "ensure_ffmpeg_env", lambda: None)
        monkeypatch.setattr(transcribe, "_detect_asr_device", lambda: "cpu")
        monkeypatch.setattr(transcribe, "_add_transcribe_record", lambda r: None)

        transcribe_file(video_file, output_dir=out_dir, keep_temp_audio=True)
        assert temp_wav.exists()


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------


class TestConstants:
    """测试模块常量。"""

    def test_default_model_is_large_v3(self):
        """默认模型是 large-v3。"""
        assert DEFAULT_ASR_MODEL == "large-v3"

    def test_supported_models_complete(self):
        """支持的模型列表完整。"""
        for m in ["tiny", "base", "small", "medium", "large-v3"]:
            assert m in SUPPORTED_MODELS

    def test_transcribe_history_file_constant(self):
        """TRANSCRIBE_HISTORY_FILE 是 Path 对象。"""
        assert isinstance(TRANSCRIBE_HISTORY_FILE, Path)
