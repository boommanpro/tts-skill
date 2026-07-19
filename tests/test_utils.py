"""测试 tts_skill.utils 工具模块。"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from unittest import mock

import pytest

from tts_skill import utils


# ---------------------------------------------------------------------------
# 平台检测
# ---------------------------------------------------------------------------


class TestPlatformDetection:
    """测试平台检测函数。"""

    def test_get_platform_returns_known_value(self):
        """get_platform 应返回 linux/macos/windows 之一。"""
        result = utils.get_platform()
        assert result in ("linux", "macos", "windows"), f"未知平台: {result}"

    def test_get_platform_macos(self):
        """模拟 macOS。"""
        with mock.patch("platform.system", return_value="Darwin"):
            assert utils.get_platform() == "macos"

    def test_get_platform_linux(self):
        """模拟 Linux。"""
        with mock.patch("platform.system", return_value="Linux"):
            assert utils.get_platform() == "linux"

    def test_get_platform_windows(self):
        """模拟 Windows。"""
        with mock.patch("platform.system", return_value="Windows"):
            assert utils.get_platform() == "windows"

    def test_is_apple_silicon_on_macos_arm(self):
        """macOS ARM 应识别为 Apple Silicon。"""
        with mock.patch("tts_skill.utils.get_platform", return_value="macos"), \
             mock.patch("platform.machine", return_value="arm64"):
            assert utils.is_apple_silicon() is True

    def test_is_apple_silicon_on_macos_intel(self):
        """macOS Intel 不应识别为 Apple Silicon。"""
        with mock.patch("tts_skill.utils.get_platform", return_value="macos"), \
             mock.patch("platform.machine", return_value="x86_64"):
            assert utils.is_apple_silicon() is False

    def test_is_apple_silicon_on_linux(self):
        """Linux 不应识别为 Apple Silicon。"""
        with mock.patch("tts_skill.utils.get_platform", return_value="linux"):
            assert utils.is_apple_silicon() is False


# ---------------------------------------------------------------------------
# 网络区域检测
# ---------------------------------------------------------------------------


class TestChinaNetwork:
    """测试国内网络检测。"""

    def test_force_cn(self, monkeypatch):
        """TTS_SKILL_FORCE_REGION=cn 应返回 True。"""
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "cn")
        assert utils.is_china_network() is True

    def test_force_global(self, monkeypatch):
        """TTS_SKILL_FORCE_REGION=global 应返回 False。"""
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "global")
        assert utils.is_china_network() is False

    def test_default_is_cn(self, monkeypatch):
        """默认（未设置）应返回 True（用户偏好国内使用）。"""
        monkeypatch.delenv("TTS_SKILL_FORCE_REGION", raising=False)
        assert utils.is_china_network() is True


# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------


class TestRandomSeed:
    """测试随机种子功能。"""

    def test_fix_random_seed_sets_random(self):
        """fix_random_seed 应设置 random 模块种子。"""
        import random

        utils.fix_random_seed(42)
        val1 = random.random()

        utils.fix_random_seed(42)
        val2 = random.random()

        assert val1 == val2, "相同种子应产生相同随机数"

    def test_fix_random_seed_different_seeds(self):
        """不同种子应产生不同结果。"""
        import random

        utils.fix_random_seed(42)
        val1 = random.random()

        utils.fix_random_seed(999)
        val2 = random.random()

        assert val1 != val2, "不同种子应产生不同随机数"

    def test_gen_random_seed_in_range(self):
        """gen_random_seed 应在合理范围内。"""
        for _ in range(100):
            seed = utils.gen_random_seed()
            assert 0 <= seed < 2**31
            assert isinstance(seed, int)

    def test_fix_random_seed_without_numpy(self):
        """没有 numpy 时 fix_random_seed 应安全降级。"""
        with mock.patch.dict("sys.modules", {"numpy": None}):
            # 不应抛异常
            utils.fix_random_seed(42)


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------


class TestPaths:
    """测试路径常量。"""

    def test_project_root_exists(self):
        """项目根目录应存在。"""
        assert utils.PROJECT_ROOT.exists()
        assert utils.PROJECT_ROOT.is_dir()

    def test_output_dir_path(self):
        """OUTPUT_DIR 应在项目根目录下。"""
        assert utils.OUTPUT_DIR.parent == utils.PROJECT_ROOT

    def test_history_file_path(self):
        """HISTORY_FILE 应在项目根目录下。"""
        assert utils.HISTORY_FILE.parent == utils.PROJECT_ROOT

    def test_setup_marker_path(self):
        """SETUP_MARKER 应在项目根目录下。"""
        assert utils.SETUP_MARKER.parent == utils.PROJECT_ROOT

    def test_ensure_output_dir(self, tmp_path, monkeypatch):
        """ensure_output_dir 应创建目录。"""
        test_dir = tmp_path / "test_outputs"
        monkeypatch.setattr(utils, "OUTPUT_DIR", test_dir)
        result = utils.ensure_output_dir()
        assert result.exists()
        assert result.is_dir()

    def test_setup_marker_lifecycle(self, tmp_path, monkeypatch):
        """测试 setup 标记的生命周期。"""
        marker = tmp_path / ".setup_done"
        monkeypatch.setattr(utils, "SETUP_MARKER", marker)

        assert utils.is_setup_done() is False
        utils.mark_setup_done()
        assert utils.is_setup_done() is True
        assert marker.exists()

        utils.clear_setup_marker()
        assert utils.is_setup_done() is False
        assert not marker.exists()


# ---------------------------------------------------------------------------
# 设备检测
# ---------------------------------------------------------------------------


class TestDeviceDetection:
    """测试设备检测。"""

    def test_detect_torch_device_without_torch(self):
        """没有 torch 时应返回 'cpu'。"""
        with mock.patch.dict("sys.modules", {"torch": None}):
            assert utils.detect_torch_device() == "cpu"

    def test_detect_torch_device_cuda(self):
        """有 CUDA 时应返回 'cuda'。"""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert utils.detect_torch_device() == "cuda"

    def test_detect_torch_device_mps(self):
        """有 MPS 时应返回 'mps'。"""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.xpu.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert utils.detect_torch_device() == "mps"

    def test_detect_torch_device_cpu(self):
        """无 GPU 时应返回 'cpu'。"""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.xpu.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert utils.detect_torch_device() == "cpu"

    def test_get_recommended_device_hint_macos_arm(self):
        """macOS ARM 应推荐 mps。"""
        with mock.patch("tts_skill.utils.get_platform", return_value="macos"), \
             mock.patch("tts_skill.utils.is_apple_silicon", return_value=True), \
             mock.patch("tts_skill.utils.has_nvidia_gpu", return_value=False):
            assert utils.get_recommended_device_hint() == "mps"

    def test_get_recommended_device_hint_with_nvidia(self):
        """有 NVIDIA GPU 应推荐 cuda。"""
        with mock.patch("tts_skill.utils.has_nvidia_gpu", return_value=True):
            assert utils.get_recommended_device_hint() == "cuda"

    def test_get_recommended_device_hint_cpu(self):
        """无 GPU 应推荐 cpu。"""
        with mock.patch("tts_skill.utils.has_nvidia_gpu", return_value=False), \
             mock.patch("tts_skill.utils.is_apple_silicon", return_value=False), \
             mock.patch("tts_skill.utils.has_intel_arc", return_value=False):
            assert utils.get_recommended_device_hint() == "cpu"


# ---------------------------------------------------------------------------
# HuggingFace 镜像
# ---------------------------------------------------------------------------


class TestHFMirror:
    """测试 HuggingFace 镜像设置。"""

    def test_set_hf_mirror_env_cn(self, monkeypatch):
        """国内环境应设置 HF_ENDPOINT。"""
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "cn")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        utils.set_hf_mirror_env()
        assert os.environ.get("HF_ENDPOINT") == utils.HF_MIRROR

    def test_set_hf_mirror_env_global(self, monkeypatch):
        """海外环境不应设置 HF_ENDPOINT。"""
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "global")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        utils.set_hf_mirror_env()
        assert "HF_ENDPOINT" not in os.environ or os.environ.get("HF_ENDPOINT") != utils.HF_MIRROR

    def test_set_hf_mirror_env_not_override(self, monkeypatch):
        """已设置的 HF_ENDPOINT 不应被覆盖。"""
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "cn")
        monkeypatch.setenv("HF_ENDPOINT", "https://custom.example.com")
        utils.set_hf_mirror_env()
        assert os.environ.get("HF_ENDPOINT") == "https://custom.example.com"


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------


class TestConstants:
    """测试常量值。"""

    def test_default_model(self):
        assert utils.DEFAULT_MODEL == "k2-fsa/OmniVoice"

    def test_hf_mirror_url(self):
        assert utils.HF_MIRROR == "https://hf-mirror.com"

    def test_pip_mirror_url(self):
        assert "aliyun" in utils.PIP_MIRROR

    def test_pytorch_cuda_index(self):
        assert "pytorch.org" in utils.PYTORCH_CUDA_INDEX

    def test_get_python_executable(self):
        result = utils.get_python_executable()
        assert isinstance(result, str)
        assert len(result) > 0
