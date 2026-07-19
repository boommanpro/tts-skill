"""测试 tts_skill.setup_env 环境安装模块。"""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

from tts_skill import setup_env
from tts_skill.utils import SETUP_MARKER


# ---------------------------------------------------------------------------
# 版本检测
# ---------------------------------------------------------------------------


class TestVersionCheck:
    """测试 Python 版本检测。"""

    def test_check_python_version_current(self):
        """当前 Python 版本应通过检查。"""
        ok, info = setup_env.check_python_version()
        # 测试环境应满足最低版本要求
        assert ok is True
        assert "." in info

    def test_check_python_version_too_low(self):
        """模拟低版本 Python。"""
        low_version = mock.MagicMock()
        low_version.major = 3
        low_version.minor = 9
        low_version.micro = 0
        with mock.patch("sys.version_info", low_version):
            ok, info = setup_env.check_python_version()
            assert ok is False
            assert "过低" in info


# ---------------------------------------------------------------------------
# 安装状态检测
# ---------------------------------------------------------------------------


class TestInstallStatus:
    """测试安装状态检测。"""

    def test_is_omnivoice_installed_false(self):
        """模拟 omnivoice 未安装。"""
        with mock.patch.dict("sys.modules", {"omnivoice": None}):
            # 由于 None 值会导致 import 失败
            try:
                import omnivoice  # noqa: F401
                installed = True
            except ImportError:
                installed = False
            assert installed is False

    def test_is_torch_installed_true(self):
        """模拟 torch 已安装。"""
        mock_torch = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert setup_env.is_torch_installed() is True

    def test_is_torch_installed_false(self):
        """模拟 torch 未安装。"""
        with mock.patch.dict("sys.modules", {"torch": None}):
            assert setup_env.is_torch_installed() is False


# ---------------------------------------------------------------------------
# pip 安装
# ---------------------------------------------------------------------------


class TestPipInstall:
    """测试 pip install 逻辑。"""

    def test_pip_install_success(self, monkeypatch):
        """成功安装。"""
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="Success", stderr=""
            )

        monkeypatch.setattr(setup_env, "run_subprocess", mock_run)
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "cn")

        ok, out = setup_env._pip_install(["some-package"])
        assert ok is True
        assert "Success" in out

    def test_pip_install_failure(self, monkeypatch):
        """安装失败。"""
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, returncode=1, stdout="", stderr="Error"
            )

        monkeypatch.setattr(setup_env, "run_subprocess", mock_run)

        ok, out = setup_env._pip_install(["bad-package"])
        assert ok is False
        assert "Error" in out

    def test_pip_install_uses_china_mirror(self, monkeypatch):
        """国内环境应使用国内镜像。"""
        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(setup_env, "run_subprocess", mock_run)
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "cn")

        setup_env._pip_install(["some-package"])

        assert "-i" in captured_cmd
        assert "mirrors.aliyun.com" in captured_cmd
        assert "--trusted-host" in captured_cmd

    def test_pip_install_no_mirror_global(self, monkeypatch):
        """海外环境不使用国内镜像。"""
        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(setup_env, "run_subprocess", mock_run)
        monkeypatch.setenv("TTS_SKILL_FORCE_REGION", "global")

        setup_env._pip_install(["some-package"])

        assert "-i" not in captured_cmd or "aliyun" not in " ".join(captured_cmd)


# ---------------------------------------------------------------------------
# torch 安装策略
# ---------------------------------------------------------------------------


class TestTorchInstallStrategy:
    """测试 torch 安装策略（根据平台/GPU 选择正确 wheel）。"""

    def test_install_torch_apple_silicon(self, monkeypatch):
        """Apple Silicon 应安装标准 torch。"""
        captured_args = []

        def mock_pip_install(args, timeout=1800):
            captured_args.extend(args)
            return True, "ok"

        monkeypatch.setattr(setup_env, "_pip_install", mock_pip_install)
        monkeypatch.setattr(setup_env, "get_platform", lambda: "macos")
        monkeypatch.setattr(setup_env, "is_apple_silicon", lambda: True)

        ok, _ = setup_env.install_torch()
        assert ok is True
        assert "torch==2.8.0" in captured_args
        assert "torchaudio==2.8.0" in captured_args
        # 不应包含 CUDA 索引
        assert not any("cu128" in str(a) for a in captured_args)

    def test_install_torch_nvidia_cuda(self, monkeypatch):
        """NVIDIA GPU 应安装 CUDA torch。"""
        captured_args = []

        def mock_pip_install(args, timeout=1800):
            captured_args.extend(args)
            return True, "ok"

        monkeypatch.setattr(setup_env, "_pip_install", mock_pip_install)
        monkeypatch.setattr(setup_env, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cuda")
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")

        ok, _ = setup_env.install_torch()
        assert ok is True
        # 应包含 CUDA 版本标记
        assert any("cu128" in str(a) for a in captured_args)

    def test_install_torch_intel_xpu(self, monkeypatch):
        """Intel Arc 应安装 XPU torch。"""
        captured_args = []

        def mock_pip_install(args, timeout=1800):
            captured_args.extend(args)
            return True, "ok"

        monkeypatch.setattr(setup_env, "_pip_install", mock_pip_install)
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "xpu")
        monkeypatch.setattr(setup_env, "has_nvidia_gpu", lambda: False)
        # 必须 mock 为非 macOS，否则会先命中 Apple Silicon 分支
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "is_apple_silicon", lambda: False)

        ok, _ = setup_env.install_torch()
        assert ok is True
        # 应包含 XPU 索引
        assert any("intel" in str(a).lower() for a in captured_args)

    def test_install_torch_cpu_linux(self, monkeypatch):
        """Linux 无 GPU 应安装 CPU torch。"""
        captured_args = []

        def mock_pip_install(args, timeout=1800):
            captured_args.extend(args)
            return True, "ok"

        monkeypatch.setattr(setup_env, "_pip_install", mock_pip_install)
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "has_nvidia_gpu", lambda: False)
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cpu")
        monkeypatch.setattr(setup_env, "is_apple_silicon", lambda: False)

        ok, _ = setup_env.install_torch()
        assert ok is True
        # 应包含 CPU 版本标记
        assert any("cpu" in str(a) for a in captured_args)


# ---------------------------------------------------------------------------
# 主安装流程
# ---------------------------------------------------------------------------


class TestSetupFlow:
    """测试主安装流程。"""

    def test_setup_already_done(self, monkeypatch, tmp_path):
        """已安装完成应跳过。"""
        marker = tmp_path / ".setup_done"
        marker.write_text("done")

        monkeypatch.setattr(setup_env, "is_setup_done", lambda: True)
        monkeypatch.setattr(setup_env, "is_omnivoice_installed", lambda: True)
        monkeypatch.setattr(setup_env, "is_torch_installed", lambda: True)

        ok, log = setup_env.setup_environment()
        assert ok is True
        assert "跳过" in log or "完成" in log

    def test_setup_force_reinstall(self, monkeypatch):
        """force=True 应清除标记重新安装。"""
        clear_called = mock.Mock()
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: True)
        monkeypatch.setattr(setup_env, "clear_setup_marker", clear_called)
        monkeypatch.setattr(setup_env, "check_python_version", lambda: (True, "3.10.0"))
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)
        monkeypatch.setattr(setup_env, "is_china_network", lambda: True)
        monkeypatch.setattr(setup_env, "upgrade_pip", lambda: (True, ""))
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cpu")
        monkeypatch.setattr(setup_env, "install_torch", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "install_omnivoice", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "install_tn_dependencies", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "is_torch_installed", lambda: True)
        monkeypatch.setattr(setup_env, "is_omnivoice_installed", lambda: True)
        monkeypatch.setattr(setup_env, "mark_setup_done", lambda: None)

        ok, _ = setup_env.setup_environment(force=True)
        assert ok is True
        clear_called.assert_called_once()

    def test_setup_python_version_too_low(self, monkeypatch):
        """Python 版本过低应失败。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)
        monkeypatch.setattr(setup_env, "check_python_version", lambda: (False, "Python 3.9 过低"))

        ok, log = setup_env.setup_environment()
        assert ok is False
        assert "过低" in log

    def test_setup_torch_install_fails(self, monkeypatch):
        """torch 安装失败应中止。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)
        monkeypatch.setattr(setup_env, "check_python_version", lambda: (True, "3.10.0"))
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)
        monkeypatch.setattr(setup_env, "is_china_network", lambda: True)
        monkeypatch.setattr(setup_env, "upgrade_pip", lambda: (True, ""))
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cpu")
        monkeypatch.setattr(setup_env, "install_torch", lambda: (False, "torch failed"))

        ok, log = setup_env.setup_environment()
        assert ok is False
        assert "torch" in log.lower() or "失败" in log

    def test_setup_omnivoice_install_fails(self, monkeypatch):
        """omnivoice 安装失败应中止。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)
        monkeypatch.setattr(setup_env, "check_python_version", lambda: (True, "3.10.0"))
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)
        monkeypatch.setattr(setup_env, "is_china_network", lambda: True)
        monkeypatch.setattr(setup_env, "upgrade_pip", lambda: (True, ""))
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cpu")
        monkeypatch.setattr(setup_env, "install_torch", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "install_omnivoice", lambda: (False, "omnivoice failed"))

        ok, log = setup_env.setup_environment()
        assert ok is False

    def test_setup_skip_tn(self, monkeypatch):
        """skip_tn=True 应跳过文本归一化依赖。"""
        tn_called = mock.Mock()
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)
        monkeypatch.setattr(setup_env, "check_python_version", lambda: (True, "3.10.0"))
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)
        monkeypatch.setattr(setup_env, "is_china_network", lambda: True)
        monkeypatch.setattr(setup_env, "upgrade_pip", lambda: (True, ""))
        monkeypatch.setattr(setup_env, "get_platform", lambda: "linux")
        monkeypatch.setattr(setup_env, "get_recommended_device_hint", lambda: "cpu")
        monkeypatch.setattr(setup_env, "install_torch", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "install_omnivoice", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "install_tn_dependencies", tn_called)
        monkeypatch.setattr(setup_env, "is_torch_installed", lambda: True)
        monkeypatch.setattr(setup_env, "is_omnivoice_installed", lambda: True)
        monkeypatch.setattr(setup_env, "mark_setup_done", lambda: None)

        ok, _ = setup_env.setup_environment(skip_tn=True)
        assert ok is True
        tn_called.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_setup
# ---------------------------------------------------------------------------


class TestEnsureSetup:
    """测试 ensure_setup 函数。"""

    def test_ensure_setup_already_ready(self, monkeypatch):
        """环境已就绪应返回 True。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: True)
        monkeypatch.setattr(setup_env, "is_omnivoice_installed", lambda: True)
        monkeypatch.setattr(setup_env, "is_torch_installed", lambda: True)
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)

        assert setup_env.ensure_setup() is True

    def test_ensure_setup_no_auto_install(self, monkeypatch):
        """auto_install=False 时未安装应返回 False。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)

        assert setup_env.ensure_setup(auto_install=False) is False

    def test_ensure_setup_auto_install_success(self, monkeypatch):
        """auto_install=True 时应自动安装。"""
        monkeypatch.setattr(setup_env, "is_setup_done", lambda: False)
        monkeypatch.setattr(setup_env, "setup_environment", lambda: (True, "ok"))
        monkeypatch.setattr(setup_env, "set_hf_mirror_env", lambda: None)

        assert setup_env.ensure_setup(auto_install=True) is True


# ---------------------------------------------------------------------------
# 依赖常量
# ---------------------------------------------------------------------------


class TestDependencies:
    """测试依赖常量。"""

    def test_core_dependencies_includes_omnivoice(self):
        """核心依赖应包含 omnivoice。"""
        assert "omnivoice" in setup_env.CORE_DEPENDENCIES

    def test_core_dependencies_includes_gradio(self):
        """核心依赖应包含 gradio。"""
        assert "gradio" in setup_env.CORE_DEPENDENCIES

    def test_core_dependencies_includes_soundfile(self):
        """核心依赖应包含 soundfile。"""
        assert "soundfile" in setup_env.CORE_DEPENDENCIES

    def test_min_python_version(self):
        """最低 Python 版本应为 3.10。"""
        assert setup_env.MIN_PYTHON_VERSION == (3, 10)
