"""测试 tts_skill.infer CLI 推理模块。"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tts_skill import infer
from tts_skill.infer import build_infer_parser, run_inference, _generate_one, _str2bool
from tts_skill.utils import OUTPUT_DIR


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


class TestInferParser:
    """测试 infer 命令参数解析器。"""

    def test_parser_basic_args(self):
        """基本参数解析。"""
        parser = build_infer_parser()
        args = parser.parse_args(["--text", "hello", "--mode", "auto"])
        assert args.text == "hello"
        assert args.mode == "auto"

    def test_parser_required_text(self):
        """--text 是必填。"""
        parser = build_infer_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_mode_choices(self):
        """mode 只接受 clone/design/auto。"""
        parser = build_infer_parser()
        args = parser.parse_args(["--text", "x", "--mode", "clone"])
        assert args.mode == "clone"

        with pytest.raises(SystemExit):
            parser.parse_args(["--text", "x", "--mode", "invalid"])

    def test_parser_clone_args(self):
        """clone 模式参数。"""
        parser = build_infer_parser()
        args = parser.parse_args([
            "--text", "hello",
            "--mode", "clone",
            "--ref_audio", "/tmp/ref.wav",
            "--ref_text", "reference",
        ])
        assert args.ref_audio == "/tmp/ref.wav"
        assert args.ref_text == "reference"

    def test_parser_design_args(self):
        """design 模式参数。"""
        parser = build_infer_parser()
        args = parser.parse_args([
            "--text", "hello",
            "--mode", "design",
            "--instruct", "male, british accent",
        ])
        assert args.instruct == "male, british accent"

    def test_parser_seed_and_batch(self):
        """种子和批量参数。"""
        parser = build_infer_parser()
        args = parser.parse_args([
            "--text", "hello",
            "--seed", "42",
            "--batch_count", "5",
        ])
        assert args.seed == 42
        assert args.batch_count == 5

    def test_parser_generation_params(self):
        """生成参数。"""
        parser = build_infer_parser()
        args = parser.parse_args([
            "--text", "hello",
            "--num_step", "16",
            "--speed", "1.5",
            "--duration", "10.0",
            "--guidance_scale", "1.5",
            "--t_shift", "0.2",
        ])
        assert args.num_step == 16
        assert args.speed == 1.5
        assert args.duration == 10.0
        assert args.guidance_scale == 1.5
        assert args.t_shift == 0.2

    def test_parser_bool_args(self):
        """布尔参数。"""
        parser = build_infer_parser()
        args = parser.parse_args([
            "--text", "hello",
            "--denoise", "false",
            "--postprocess_output", "false",
            "--normalize_text", "true",
        ])
        assert args.denoise is False
        assert args.postprocess_output is False
        assert args.normalize_text is True

    def test_str2bool_true_values(self):
        """str2bool 正确值。"""
        for v in ("yes", "true", "t", "y", "1", "True", "YES"):
            assert _str2bool(v) is True

    def test_str2bool_false_values(self):
        """str2bool 错误值。"""
        for v in ("no", "false", "f", "n", "0", "False", "NO"):
            assert _str2bool(v) is False

    def test_str2bool_invalid(self):
        """str2bool 无效值应抛异常。"""
        with pytest.raises(argparse.ArgumentTypeError):
            _str2bool("invalid")


# ---------------------------------------------------------------------------
# _generate_one 单次生成逻辑
# ---------------------------------------------------------------------------


class TestGenerateOne:
    """测试 _generate_one 函数。"""

    def test_generate_auto_mode(self):
        """auto 模式生成。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        audio = _generate_one(
            mock_model,
            text="hello",
            mode="auto",
            ref_audio=None,
            ref_text=None,
            instruct=None,
            language=None,
            seed=42,
            num_step=32,
            speed=1.0,
            duration=None,
            guidance_scale=2.0,
            t_shift=0.1,
            denoise=True,
            postprocess_output=True,
            normalize_text=False,
        )

        assert audio is not None
        assert len(audio) > 0
        mock_model.generate.assert_called_once()

    def test_generate_clone_mode(self):
        """clone 模式应传递 ref_audio。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        _generate_one(
            mock_model,
            text="hello",
            mode="clone",
            ref_audio="/tmp/ref.wav",
            ref_text="reference",
            instruct=None,
            language=None,
            seed=42,
            num_step=32,
            speed=1.0,
            duration=None,
            guidance_scale=2.0,
            t_shift=0.1,
            denoise=True,
            postprocess_output=True,
            normalize_text=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["ref_audio"] == "/tmp/ref.wav"
        assert call_kwargs["ref_text"] == "reference"

    def test_generate_design_mode(self):
        """design 模式应传递 instruct。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        _generate_one(
            mock_model,
            text="hello",
            mode="design",
            ref_audio=None,
            ref_text=None,
            instruct="male, british accent",
            language=None,
            seed=42,
            num_step=32,
            speed=1.0,
            duration=None,
            guidance_scale=2.0,
            t_shift=0.1,
            denoise=True,
            postprocess_output=True,
            normalize_text=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["instruct"] == "male, british accent"

    def test_generate_with_speed(self):
        """speed != 1.0 应传递 speed。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        _generate_one(
            mock_model,
            text="hello",
            mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.5, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["speed"] == 1.5

    def test_generate_speed_1_not_passed(self):
        """speed == 1.0 不应传递 speed 参数。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        _generate_one(
            mock_model,
            text="hello",
            mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert "speed" not in call_kwargs

    def test_generate_with_duration(self):
        """设置 duration 应传递。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        _generate_one(
            mock_model,
            text="hello",
            mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=5.0,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["duration"] == 5.0

    def test_generate_calls_fix_random_seed(self):
        """应调用 fix_random_seed。"""
        mock_model = mock.MagicMock()
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        with mock.patch("tts_skill.infer.fix_random_seed") as mock_seed:
            _generate_one(
                mock_model,
                text="hello",
                mode="auto",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                seed=42, num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
            )
            mock_seed.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# run_inference 端到端逻辑（mock 模型）
# ---------------------------------------------------------------------------


class TestRunInference:
    """测试 run_inference 主流程。"""

    def test_infer_clone_without_ref_audio_fails(self, monkeypatch):
        """clone 模式无 ref_audio 应失败。"""
        args = argparse.Namespace(
            text="hello", mode="clone",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=None, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=1, output_dir=str(OUTPUT_DIR), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )
        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)

        result = run_inference(args)
        assert result == 1

    def test_infer_design_without_instruct_fails(self, monkeypatch):
        """design 模式无 instruct 应失败。"""
        args = argparse.Namespace(
            text="hello", mode="design",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=None, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=1, output_dir=str(OUTPUT_DIR), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )
        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)

        result = run_inference(args)
        assert result == 1

    def test_infer_setup_not_done(self, monkeypatch):
        """环境未安装应失败。"""
        args = argparse.Namespace(
            text="hello", mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=None, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=1, output_dir=str(OUTPUT_DIR), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )
        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: False)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)

        result = run_inference(args)
        assert result == 1

    def test_infer_auto_success(self, monkeypatch, tmp_path):
        """auto 模式成功生成。"""
        # 准备 mock 模型
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]
        mock_model.device = "cpu"

        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)
        monkeypatch.setattr("tts_skill.infer.detect_torch_device", lambda: "cpu")
        monkeypatch.setattr("tts_skill.infer._load_model", lambda *a, **k: mock_model)
        monkeypatch.setattr("tts_skill.infer.ensure_output_dir", lambda: tmp_path)

        # 隔离历史记录
        from tts_skill import config
        monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.json")
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        args = argparse.Namespace(
            text="hello world", mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=1, output_dir=str(tmp_path), name="test_run",
            model="k2-fsa/OmniVoice", device=None,
        )

        result = run_inference(args)
        assert result == 0

        # 验证音频文件已生成
        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 1

        # 验证历史记录已写入
        assert (tmp_path / "history.json").exists()

    def test_infer_batch_success(self, monkeypatch, tmp_path):
        """批量生成成功。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]
        mock_model.device = "cpu"

        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)
        monkeypatch.setattr("tts_skill.infer.detect_torch_device", lambda: "cpu")
        monkeypatch.setattr("tts_skill.infer._load_model", lambda *a, **k: mock_model)
        monkeypatch.setattr("tts_skill.infer.ensure_output_dir", lambda: tmp_path)

        from tts_skill import config
        monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.json")
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        args = argparse.Namespace(
            text="hello", mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=3, output_dir=str(tmp_path), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )

        result = run_inference(args)
        assert result == 0

        # 应生成 3 个音频文件
        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 3

    def test_infer_seed_reproducibility(self, monkeypatch, tmp_path):
        """相同种子应调用 fix_random_seed。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]
        mock_model.device = "cpu"

        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)
        monkeypatch.setattr("tts_skill.infer.detect_torch_device", lambda: "cpu")
        monkeypatch.setattr("tts_skill.infer._load_model", lambda *a, **k: mock_model)
        monkeypatch.setattr("tts_skill.infer.ensure_output_dir", lambda: tmp_path)

        from tts_skill import config
        monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.json")
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        seed_call_count = mock.Mock()
        monkeypatch.setattr("tts_skill.infer.fix_random_seed", seed_call_count)

        args = argparse.Namespace(
            text="hello", mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=100, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=2, output_dir=str(tmp_path), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )

        result = run_inference(args)
        assert result == 0
        # 批量 2 个，应调用 2 次 fix_random_seed
        assert seed_call_count.call_count == 2

    def test_infer_output_reproducible_command(self, monkeypatch, tmp_path, capsys):
        """应输出可复现命令。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]
        mock_model.device = "cpu"

        monkeypatch.setattr("tts_skill.infer.is_setup_done", lambda: True)
        monkeypatch.setattr("tts_skill.infer.set_hf_mirror_env", lambda: None)
        monkeypatch.setattr("tts_skill.infer.detect_torch_device", lambda: "cpu")
        monkeypatch.setattr("tts_skill.infer._load_model", lambda *a, **k: mock_model)
        monkeypatch.setattr("tts_skill.infer.ensure_output_dir", lambda: tmp_path)

        from tts_skill import config
        monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.json")
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        args = argparse.Namespace(
            text="hello world", mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
            batch_count=1, output_dir=str(tmp_path), name=None,
            model="k2-fsa/OmniVoice", device=None,
        )

        result = run_inference(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "可复现命令" in captured.out
        assert "tts_skill" in captured.out
        assert "infer" in captured.out
        assert "--seed" in captured.out
        assert "42" in captured.out
