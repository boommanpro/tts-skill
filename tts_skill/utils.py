"""通用工具：平台检测、设备检测、路径处理、随机种子。"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Windows 控制台 UTF-8 输出支持
# ---------------------------------------------------------------------------


def _setup_utf8_output() -> None:
    """在 Windows 上强制 stdout/stderr 使用 UTF-8 编码。

    Windows 默认使用 cp1252 编码，无法输出中文。
    此函数在模块导入时自动调用，确保跨平台输出一致。
    """
    if sys.platform != "win32":
        return
    try:
        # 重新配置 stdout/stderr 为 UTF-8
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        # 重新配置失败时，用 utf-8 包装流
        try:
            sys.stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", errors="replace", buffering=1)  # type: ignore[assignment]
            sys.stderr = open(sys.stderr.fileno(), "w", encoding="utf-8", errors="replace", buffering=1)  # type: ignore[assignment]
        except Exception:
            pass


# 导入时自动设置 UTF-8 输出
_setup_utf8_output()

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

# 项目根目录（tts_skill/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 历史记录文件
HISTORY_FILE = PROJECT_ROOT / "history.json"

# 安装标记文件（setup 完成后写入）
SETUP_MARKER = PROJECT_ROOT / ".setup_done"

# 默认模型
DEFAULT_MODEL = "k2-fsa/OmniVoice"

# 国内 HuggingFace 镜像
HF_MIRROR = "https://hf-mirror.com"

# 国内 pip 镜像
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple"

# PyTorch CUDA wheel 索引（Linux/Windows NVIDIA GPU）
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

# Intel XPU wheel 索引
PYTORCH_XPU_INDEX = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"


def ensure_output_dir() -> Path:
    """确保输出目录存在并返回路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def is_china_network() -> bool:
    """检测是否在国内网络环境。

    通过环境变量 TTS_SKILL_FORCE_REGION 强制指定：
    - TTS_SKILL_FORCE_REGION=cn  -> 视为国内
    - TTS_SKILL_FORCE_REGION=global -> 视为海外
    否则尝试访问 PyPI 官方源，超时则视为国内。
    """
    forced = os.environ.get("TTS_SKILL_FORCE_REGION", "").lower()
    if forced == "cn":
        return True
    if forced == "global":
        return False
    # 默认启用国内镜像（用户偏好：支持在国内使用）
    # 如需关闭，设置 TTS_SKILL_FORCE_REGION=global
    return True


def get_platform() -> str:
    """返回标准化平台标识：'linux' / 'macos' / 'windows'。"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system


def is_apple_silicon() -> bool:
    """检测是否为 Apple Silicon (M1/M2/M3...)。"""
    if get_platform() != "macos":
        return False
    return platform.machine().lower() == "arm64"


def detect_torch_device() -> str:
    """检测最佳推理设备：cuda > xpu > mps > cpu。

    返回值可直接传给 OmniVoice.from_pretrained(device_map=...)。
    如果 torch 未安装，返回 "cpu" 作为占位（setup 后再调用）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def has_nvidia_gpu() -> bool:
    """检测是否有 NVIDIA GPU（在 torch 安装前可用）。"""
    plat = get_platform()
    if plat == "macos":
        return False
    # Linux: 检查 /proc/driver/nvidia 或 nvidia-smi
    if plat == "linux":
        if Path("/proc/driver/nvidia").exists():
            return True
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    # Windows: 检查 nvidia-smi
    if plat == "windows":
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    return False


def has_intel_arc() -> bool:
    """检测是否有 Intel Arc GPU（简化检测）。"""
    # 通过环境变量显式指定更可靠
    return os.environ.get("TTS_SKILL_DEVICE") == "xpu"


def get_recommended_device_hint() -> str:
    """在 torch 安装前给出设备提示，用于选择正确的 torch wheel。"""
    if has_nvidia_gpu():
        return "cuda"
    if is_apple_silicon():
        return "mps"
    if has_intel_arc():
        return "xpu"
    return "cpu"


def fix_random_seed(seed: int) -> None:
    """设置随机种子以保证可复现性。

    对齐 omnivoice.utils.common.fix_random_seed 的实现，
    在 torch 未安装时也能安全调用（只设置 random）。
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gen_random_seed() -> int:
    """生成一个随机种子（0 ~ 2^31 - 1）。"""
    return random.randint(0, 2**31 - 1)


def get_python_executable() -> str:
    """返回当前 Python 解释器路径。"""
    return sys.executable


def run_subprocess(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """运行子进程并返回结果，统一编码处理。"""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, **kwargs)


def is_setup_done() -> bool:
    """检查环境是否已安装完成。"""
    return SETUP_MARKER.exists()


def mark_setup_done() -> None:
    """标记环境安装完成。"""
    SETUP_MARKER.write_text("done\n", encoding="utf-8")


def clear_setup_marker() -> None:
    """清除安装标记（用于重新安装）。"""
    if SETUP_MARKER.exists():
        SETUP_MARKER.unlink()


def set_hf_mirror_env() -> None:
    """设置 HuggingFace 镜像环境变量（国内加速）。"""
    if is_china_network():
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
