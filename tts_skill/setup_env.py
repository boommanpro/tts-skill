"""跨平台环境自动安装。

检测平台与 GPU，使用国内镜像安装 PyTorch + OmniVoice 及所有依赖。
支持 Linux / macOS / Windows，支持 NVIDIA CUDA / Apple MPS / Intel XPU / CPU。
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from tts_skill.utils import (
    PIP_MIRROR,
    PYTORCH_CUDA_INDEX,
    PYTORCH_XPU_INDEX,
    clear_setup_marker,
    get_platform,
    get_python_executable,
    get_recommended_device_hint,
    has_nvidia_gpu,
    is_apple_silicon,
    is_china_network,
    is_setup_done,
    mark_setup_done,
    run_subprocess,
    set_hf_mirror_env,
)

logger = logging.getLogger(__name__)

# OmniVoice 最低 Python 版本
MIN_PYTHON_VERSION = (3, 10)

# 核心 pip 依赖（除 torch 外）
CORE_DEPENDENCIES = [
    "omnivoice",
    "soundfile",
    "numpy",
    "gradio",
]

# 语音转文字（ASR）依赖
# - faster-whisper: 基于 CTranslate2 的 Whisper 加速实现，不依赖 torch
# - imageio-ffmpeg: 提供跨平台 ffmpeg 二进制，用于视频提取音轨（无需用户手动装 ffmpeg）
ASR_DEPENDENCIES = [
    "faster-whisper>=1.0.0",
    "imageio-ffmpeg>=0.4.9",
]

# 可选：文本归一化依赖（国内 macOS 无 pynini wheel，跳过）
TN_DEPENDENCIES = ["num2words"]


# ---------------------------------------------------------------------------
# 版本检测
# ---------------------------------------------------------------------------


def check_python_version() -> tuple[bool, str]:
    """检查 Python 版本是否满足要求。"""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON_VERSION:
        return False, f"Python {version_str} 过低，需要 >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
    return True, version_str


def is_omnivoice_installed() -> bool:
    """检查 omnivoice 是否已安装。"""
    try:
        import omnivoice  # noqa: F401

        return True
    except ImportError:
        return False


def is_torch_installed() -> bool:
    """检查 torch 是否已安装。"""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def is_asr_installed() -> bool:
    """检查语音转文字依赖是否已安装（faster-whisper + imageio-ffmpeg）。"""
    try:
        import faster_whisper  # noqa: F401

        import imageio_ffmpeg  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# pip 安装
# ---------------------------------------------------------------------------


def _pip_install(args: list[str], timeout: int = 1800) -> tuple[bool, str]:
    """执行 pip install 命令，返回 (成功与否, 输出日志)。"""
    cmd = [get_python_executable(), "-m", "pip", "install"] + args

    # 国内镜像加速
    if is_china_network():
        cmd += ["-i", PIP_MIRROR, "--trusted-host", "mirrors.aliyun.com"]

    logger.info("运行: %s", " ".join(cmd))
    result = run_subprocess(cmd, timeout=timeout)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return False, output
    return True, output


def install_torch() -> tuple[bool, str]:
    """根据平台和 GPU 安装对应版本的 PyTorch。

    返回 (成功与否, 日志)。
    """
    plat = get_platform()
    device_hint = get_recommended_device_hint()

    # Apple Silicon (macOS ARM): 标准 PyTorch（含 MPS 支持）
    if plat == "macos" and is_apple_silicon():
        logger.info("安装 PyTorch (Apple Silicon / MPS)...")
        return _pip_install(
            ["torch==2.8.0", "torchaudio==2.8.0"],
            timeout=1800,
        )

    # macOS Intel: 标准 PyTorch（CPU）
    if plat == "macos":
        logger.info("安装 PyTorch (macOS Intel / CPU)...")
        return _pip_install(
            ["torch==2.8.0", "torchaudio==2.8.0"],
            timeout=1800,
        )

    # Linux/Windows + NVIDIA GPU: CUDA 版本
    if device_hint == "cuda" or has_nvidia_gpu():
        logger.info("安装 PyTorch (NVIDIA CUDA cu128)...")
        # 使用额外索引（PyTorch 官方源），主源用国内镜像
        if is_china_network():
            ok, out = _pip_install(
                [
                    "torch==2.8.0+cu128",
                    "torchaudio==2.8.0+cu128",
                    "--extra-index-url",
                    PYTORCH_CUDA_INDEX,
                ],
                timeout=1800,
            )
        else:
            ok, out = _pip_install(
                [
                    "torch==2.8.0+cu128",
                    "torchaudio==2.8.0+cu128",
                    "--extra-index-url",
                    PYTORCH_CUDA_INDEX,
                ],
                timeout=1800,
            )
        return ok, out

    # Intel Arc XPU
    if device_hint == "xpu":
        logger.info("安装 PyTorch (Intel XPU)...")
        return _pip_install(
            [
                "torch",
                "torchaudio",
                "--index-url",
                PYTORCH_XPU_INDEX,
            ],
            timeout=1800,
        )

    # 默认 CPU 版本（Linux/Windows 无 GPU）
    logger.info("安装 PyTorch (CPU)...")
    if plat in ("linux", "windows"):
        # CPU 版本使用 PyTorch 官方 CPU 索引
        return _pip_install(
            [
                "torch==2.8.0+cpu",
                "torchaudio==2.8.0+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            timeout=1800,
        )
    return _pip_install(["torch==2.8.0", "torchaudio==2.8.0"], timeout=1800)


def install_omnivoice() -> tuple[bool, str]:
    """安装 OmniVoice 及其依赖。"""
    logger.info("安装 OmniVoice...")
    return _pip_install(CORE_DEPENDENCIES, timeout=1800)


def install_asr() -> tuple[bool, str]:
    """安装语音转文字（ASR）依赖：faster-whisper + imageio-ffmpeg。"""
    logger.info("安装语音转文字依赖...")
    return _pip_install(ASR_DEPENDENCIES, timeout=600)


def install_tn_dependencies() -> tuple[bool, str]:
    """安装文本归一化可选依赖。"""
    logger.info("安装文本归一化依赖...")
    return _pip_install(TN_DEPENDENCIES, timeout=600)


def upgrade_pip() -> tuple[bool, str]:
    """升级 pip 以避免旧版本兼容性问题。"""
    logger.info("升级 pip...")
    return _pip_install(["--upgrade", "pip"], timeout=300)


# ---------------------------------------------------------------------------
# 主安装流程
# ---------------------------------------------------------------------------


def setup_environment(force: bool = False, skip_tn: bool = False, skip_asr: bool = False) -> tuple[bool, str]:
    """执行完整环境安装流程。

    Args:
        force: 强制重新安装（忽略已有标记）。
        skip_tn: 跳过文本归一化可选依赖。
        skip_asr: 跳过语音转文字（ASR）依赖。

    Returns:
        (成功与否, 日志信息)
    """
    all_logs: list[str] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        all_logs.append(msg)

    # 1. 检查 Python 版本
    ok, info = check_python_version()
    if not ok:
        _log(f"[错误] {info}")
        return False, "\n".join(all_logs)
    _log(f"[1/6] Python 版本检查通过: {info}")

    # 2. 检查是否已安装
    if is_setup_done() and not force:
        if is_omnivoice_installed() and is_torch_installed():
            _log("[完成] 环境已安装，跳过。使用 --force 可重新安装。")
            return True, "\n".join(all_logs)

    if force:
        clear_setup_marker()

    # 3. 设置国内镜像环境变量
    set_hf_mirror_env()
    _log(f"[2/6] HuggingFace 镜像: {('启用 ' + 'https://hf-mirror.com') if is_china_network() else '未启用'}")
    _log(f"      pip 镜像: {'启用 ' + PIP_MIRROR if is_china_network() else '未启用'}")

    # 4. 升级 pip
    _log("[3/6] 升级 pip...")
    ok, out = upgrade_pip()
    if not ok:
        _log("[警告] pip 升级失败，继续安装...")

    # 5. 安装 PyTorch
    _log("[4/6] 安装 PyTorch（根据平台和 GPU 自动选择）...")
    device_hint = get_recommended_device_hint()
    plat = get_platform()
    _log(f"      平台: {plat}, 设备: {device_hint}")
    ok, out = install_torch()
    if not ok:
        _log("[错误] PyTorch 安装失败:")
        _log(out[-2000:] if len(out) > 2000 else out)
        return False, "\n".join(all_logs)
    _log("[4/6] PyTorch 安装成功")

    # 6. 安装 OmniVoice
    _log("[5/6] 安装 OmniVoice...")
    ok, out = install_omnivoice()
    if not ok:
        _log("[错误] OmniVoice 安装失败:")
        _log(out[-2000:] if len(out) > 2000 else out)
        return False, "\n".join(all_logs)
    _log("[5/6] OmniVoice 安装成功")

    # 7. 安装语音转文字（ASR）依赖
    if not skip_asr:
        _log("[6/6] 安装语音转文字依赖（faster-whisper + imageio-ffmpeg）...")
        ok, out = install_asr()
        if ok:
            _log("[6/6] 语音转文字依赖安装成功")
        else:
            _log("[警告] 语音转文字依赖安装失败（不影响 TTS 核心功能）:")
            _log(out[-1000:] if len(out) > 1000 else out)
    else:
        _log("[6/6] 跳过语音转文字依赖（--skip_asr）")

    # 8. 可选：文本归一化依赖
    if not skip_tn:
        _log("[可选] 安装文本归一化依赖...")
        ok, _ = install_tn_dependencies()
        if ok:
            _log("[可选] 文本归一化依赖安装成功")
        else:
            _log("[可选] 文本归一化依赖安装失败（不影响核心功能）")

    # 9. 验证安装
    _log("[验证] 检查安装结果...")
    if not is_torch_installed():
        _log("[错误] torch 未正确安装")
        return False, "\n".join(all_logs)
    if not is_omnivoice_installed():
        _log("[错误] omnivoice 未正确安装")
        return False, "\n".join(all_logs)
    if not is_asr_installed():
        _log("[警告] 语音转文字依赖未正确安装（transcribe 命令不可用，TTS 功能不受影响）")

    # 输出设备信息
    try:
        import torch

        device_info = f"torch={torch.__version__}, "
        if torch.cuda.is_available():
            device_info += f"cuda={torch.cuda.get_device_name(0)}"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_info += "xpu=Intel Arc"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info += "mps=Apple Silicon"
        else:
            device_info += "device=cpu"
        _log(f"[验证] {device_info}")
    except Exception as e:
        _log(f"[警告] 无法获取设备信息: {e}")

    mark_setup_done()
    _log("[完成] 环境安装完成！")
    return True, "\n".join(all_logs)


def ensure_setup(auto_install: bool = True) -> bool:
    """确保环境已安装，未安装时自动安装。

    Args:
        auto_install: 是否自动安装（False 时只检查不安装）。

    Returns:
        环境是否就绪。
    """
    if is_setup_done() and is_omnivoice_installed() and is_torch_installed():
        set_hf_mirror_env()
        return True

    if not auto_install:
        return False

    ok, _ = setup_environment()
    set_hf_mirror_env()
    return ok
