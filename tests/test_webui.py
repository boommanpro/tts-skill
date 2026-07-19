"""测试 tts_skill.webui Web UI 模块。"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from tts_skill import webui


# ---------------------------------------------------------------------------
# 声音设计属性构建
# ---------------------------------------------------------------------------


class TestBuildInstruct:
    """测试 _build_instruct_from_selections 函数。"""

    def test_empty_selections(self):
        """空选择应返回 None。"""
        result = webui._build_instruct_from_selections([])
        assert result is None

    def test_all_auto(self):
        """全部 Auto 应返回 None。"""
        result = webui._build_instruct_from_selections(["Auto", "Auto", "Auto"])
        assert result is None

    def test_single_gender(self):
        """单个性别选择。"""
        result = webui._build_instruct_from_selections(["男 / Male"])
        assert result == "Male"

    def test_multiple_selections(self):
        """多个选择应组合。"""
        result = webui._build_instruct_from_selections(
            ["男 / Male", "青年 / Young Adult", "低音调 / Low Pitch"]
        )
        assert "Male" in result
        assert "Young Adult" in result
        assert "Low Pitch" in result

    def test_dialect_uses_chinese(self):
        """方言应使用中文。"""
        result = webui._build_instruct_from_selections(["四川话"])
        assert "四川话" in result

    def test_english_accent(self):
        """英文口音应使用英文。"""
        result = webui._build_instruct_from_selections(["美式口音 / American Accent"])
        assert "American Accent" in result

    def test_mixed_auto_and_selection(self):
        """混合 Auto 和选择。"""
        result = webui._build_instruct_from_selections(["Auto", "男 / Male", "Auto"])
        assert result == "Male"


# ---------------------------------------------------------------------------
# 设计类别常量
# ---------------------------------------------------------------------------


class TestDesignCategories:
    """测试声音设计类别常量。"""

    def test_has_gender_category(self):
        assert "性别 / Gender" in webui._DESIGN_CATEGORIES

    def test_has_age_category(self):
        assert "年龄 / Age" in webui._DESIGN_CATEGORIES

    def test_has_pitch_category(self):
        assert "音调 / Pitch" in webui._DESIGN_CATEGORIES

    def test_has_style_category(self):
        assert "风格 / Style" in webui._DESIGN_CATEGORIES

    def test_has_accent_category(self):
        assert "英文口音 / English Accent" in webui._DESIGN_CATEGORIES

    def test_has_dialect_category(self):
        assert "中文方言 / Chinese Dialect" in webui._DESIGN_CATEGORIES

    def test_all_categories_start_with_auto(self):
        """所有类别都应包含 Auto 选项。"""
        for cat, choices in webui._DESIGN_CATEGORIES.items():
            assert "Auto" in choices, f"{cat} 缺少 Auto 选项"

    def test_gender_has_male_female(self):
        choices = webui._DESIGN_CATEGORIES["性别 / Gender"]
        assert any("Male" in c for c in choices)
        assert any("Female" in c for c in choices)

    def test_dialect_has_sichuan(self):
        choices = webui._DESIGN_CATEGORIES["中文方言 / Chinese Dialect"]
        assert "四川话" in choices


# ---------------------------------------------------------------------------
# 生成核心逻辑（mock 模型）
# ---------------------------------------------------------------------------


class TestGenerateAudio:
    """测试 _generate_audio 函数。"""

    def test_generate_auto_mode(self):
        """auto 模式生成。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        audio, sr = webui._generate_audio(
            mock_model,
            text="hello",
            mode="auto",
            ref_audio=None, ref_text=None, instruct=None, language=None,
            seed=42, num_step=32, speed=1.0, duration=None,
            guidance_scale=2.0, t_shift=0.1,
            denoise=True, postprocess_output=True, normalize_text=False,
        )

        assert sr == 24000
        assert len(audio) > 0

    def test_generate_clone_requires_ref_audio(self):
        """clone 模式无 ref_audio 应抛异常。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000

        with pytest.raises(ValueError, match="参考音频"):
            webui._generate_audio(
                mock_model,
                text="hello",
                mode="clone",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                seed=42, num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
            )

    def test_generate_design_requires_instruct(self):
        """design 模式无 instruct 应抛异常。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000

        with pytest.raises(ValueError, match="声音属性"):
            webui._generate_audio(
                mock_model,
                text="hello",
                mode="design",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                seed=42, num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
            )

    def test_generate_calls_fix_random_seed(self):
        """应调用 fix_random_seed。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        with mock.patch("tts_skill.webui.fix_random_seed") as mock_seed:
            webui._generate_audio(
                mock_model,
                text="hello",
                mode="auto",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                seed=42, num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
            )
            mock_seed.assert_called_once_with(42)

    def test_generate_with_duration(self):
        """设置 duration 应传递。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        webui._generate_audio(
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


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------


class TestBatchGenerate:
    """测试 _do_batch_generate 函数。"""

    def test_batch_generate_success(self, tmp_path):
        """批量生成成功。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.device = "cpu"
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        # 隔离历史记录
        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "history.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            record, output_files, audio_samples, errors = webui._do_batch_generate(
                mock_model,
                text="hello",
                mode="auto",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                base_seed=42,
                batch_count=3,
                num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
                output_dir=tmp_path,
            )

        assert len(output_files) == 3
        assert len(audio_samples) == 3
        assert len(errors) == 0
        assert record.batch_count == 3
        assert record.batch_seeds == [42, 43, 44]

    def test_batch_generate_partial_failure(self, tmp_path):
        """部分生成失败仍应返回成功的结果。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.device = "cpu"
        # 第一次成功，第二次失败，第三次成功
        mock_model.generate.side_effect = [
            [np.zeros(24000, dtype=np.float32)],
            RuntimeError("fail"),
            [np.zeros(24000, dtype=np.float32)],
        ]

        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "history.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            record, output_files, audio_samples, errors = webui._do_batch_generate(
                mock_model,
                text="hello",
                mode="auto",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                base_seed=42,
                batch_count=3,
                num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
                output_dir=tmp_path,
            )

        assert len(output_files) == 2  # 只有 2 个成功
        assert len(errors) == 1  # 1 个失败

    def test_batch_generate_all_fail(self, tmp_path):
        """全部失败应抛异常。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.device = "cpu"
        mock_model.generate.side_effect = RuntimeError("fail")

        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "history.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="所有生成均失败"):
                webui._do_batch_generate(
                    mock_model,
                    text="hello",
                    mode="auto",
                    ref_audio=None, ref_text=None, instruct=None, language=None,
                    base_seed=42,
                    batch_count=2,
                    num_step=32, speed=1.0, duration=None,
                    guidance_scale=2.0, t_shift=0.1,
                    denoise=True, postprocess_output=True, normalize_text=False,
                    output_dir=tmp_path,
                )

    def test_batch_seeds_increment(self, tmp_path):
        """批量种子应递增。"""
        mock_model = mock.MagicMock()
        mock_model.sampling_rate = 24000
        mock_model.device = "cpu"
        mock_model.generate.return_value = [np.zeros(24000, dtype=np.float32)]

        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "history.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            record, _, _, _ = webui._do_batch_generate(
                mock_model,
                text="hello",
                mode="auto",
                ref_audio=None, ref_text=None, instruct=None, language=None,
                base_seed=100,
                batch_count=5,
                num_step=32, speed=1.0, duration=None,
                guidance_scale=2.0, t_shift=0.1,
                denoise=True, postprocess_output=True, normalize_text=False,
                output_dir=tmp_path,
            )

        assert record.batch_seeds == [100, 101, 102, 103, 104]


# ---------------------------------------------------------------------------
# 历史记录表格数据
# ---------------------------------------------------------------------------


class TestHistoryTable:
    """测试历史记录表格数据生成。"""

    def test_empty_history(self, tmp_path):
        """空历史应返回空列表（list 格式）。"""
        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "empty.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            data = webui._history_table_data()
        assert data == []

    def test_empty_history_markdown(self, tmp_path):
        """空历史 Markdown 应返回提示文本。"""
        from tts_skill import config
        with mock.patch.object(config, "HISTORY_FILE", tmp_path / "empty.json"), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            md = webui._history_markdown()
        assert "暂无" in md

    def test_history_with_records(self, tmp_path):
        """有记录应返回表格行（list 格式）。"""
        from tts_skill import config
        history_file = tmp_path / "history.json"
        with mock.patch.object(config, "HISTORY_FILE", history_file), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            from tts_skill.config import create_record

            create_record(
                mode="auto",
                text="hello",
                seed=42,
                batch_count=1,
                output_files=["/tmp/test.wav"],
                name="test",
            )

            data = webui._history_table_data()
            assert len(data) == 1
            row = data[0]
            assert row[1] == "test"  # name
            assert row[2] == "auto"  # mode
            assert row[4] == "42"  # seed
            assert row[5] == "1"  # batch_count

    def test_history_with_records_markdown(self, tmp_path):
        """有记录 Markdown 应包含表格。"""
        from tts_skill import config
        history_file = tmp_path / "history.json"
        with mock.patch.object(config, "HISTORY_FILE", history_file), \
             mock.patch.object(config, "OUTPUT_DIR", tmp_path):
            from tts_skill.config import create_record

            create_record(
                mode="auto",
                text="hello",
                seed=42,
                batch_count=1,
                output_files=["/tmp/test.wav"],
                name="测试记录",
            )

            md = webui._history_markdown()
            assert "|" in md  # Markdown 表格语法
            assert "测试记录" in md
            assert "42" in md


# ---------------------------------------------------------------------------
# 模型缓存
# ---------------------------------------------------------------------------


class TestModelCache:
    """测试模型缓存机制。"""

    def test_get_model_caches(self):
        """相同参数应使用缓存。"""
        webui._MODEL_CACHE.clear()

        mock_model = mock.MagicMock()
        mock_omnivoice = mock.MagicMock()
        mock_omnivoice.OmniVoice.from_pretrained.return_value = mock_model
        mock_torch = mock.MagicMock()
        mock_torch.float16 = "float16"

        with mock.patch.dict("sys.modules", {
            "torch": mock_torch,
            "omnivoice": mock_omnivoice,
        }), mock.patch("tts_skill.webui.detect_torch_device", lambda: "cpu"):
            m1 = webui._get_model("test-model", "cpu")
            m2 = webui._get_model("test-model", "cpu")

            assert m1 is m2
            # from_pretrained 只应调用一次
            assert mock_omnivoice.OmniVoice.from_pretrained.call_count == 1

    def test_get_model_different_keys(self):
        """不同参数应加载不同模型。"""
        webui._MODEL_CACHE.clear()

        mock_model1 = mock.MagicMock()
        mock_model2 = mock.MagicMock()
        mock_omnivoice = mock.MagicMock()
        mock_omnivoice.OmniVoice.from_pretrained.side_effect = [mock_model1, mock_model2]
        mock_torch = mock.MagicMock()
        mock_torch.float16 = "float16"

        with mock.patch.dict("sys.modules", {
            "torch": mock_torch,
            "omnivoice": mock_omnivoice,
        }), mock.patch("tts_skill.webui.detect_torch_device", lambda: "cpu"):
            m1 = webui._get_model("model-a", "cpu")
            m2 = webui._get_model("model-b", "cpu")

            assert m1 is not m2
            assert mock_omnivoice.OmniVoice.from_pretrained.call_count == 2


# ---------------------------------------------------------------------------
# Web UI 构建（不实际启动）
# ---------------------------------------------------------------------------


class TestWebUIBuild:
    """测试 Web UI 构建（不实际启动服务器）。"""

    def test_build_demo_returns_blocks(self):
        """build_demo 应返回 gr.Blocks 对象。"""
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("gradio 未安装，跳过 Web UI 构建测试")

        demo = webui.build_demo(model_name="test-model", device="cpu")
        assert isinstance(demo, gr.Blocks)

    def test_build_webui_parser(self):
        """build_webui_parser 应返回正确的参数解析器。"""
        parser = webui.build_webui_parser()
        args = parser.parse_args([])
        assert args.model == "k2-fsa/OmniVoice"
        assert args.port == 7860
        assert args.ip == "0.0.0.0"

    def test_build_webui_parser_custom_args(self):
        """自定义参数。"""
        parser = webui.build_webui_parser()
        args = parser.parse_args([
            "--port", "9000",
            "--ip", "127.0.0.1",
            "--share",
        ])
        assert args.port == 9000
        assert args.ip == "127.0.0.1"
        assert args.share is True


# ---------------------------------------------------------------------------
# run_webui 入口
# ---------------------------------------------------------------------------


class TestRunWebui:
    """测试 run_webui 函数。"""

    def test_run_webui_not_setup(self, monkeypatch):
        """环境未安装应返回 1。"""
        monkeypatch.setattr("tts_skill.webui.is_setup_done", lambda: False)
        monkeypatch.setattr("tts_skill.webui.set_hf_mirror_env", lambda: None)

        result = webui.run_webui()
        assert result == 1
