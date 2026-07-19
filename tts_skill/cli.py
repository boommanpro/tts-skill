"""tts-skill 主 CLI 入口。

子命令：
    setup       - 安装环境（跨平台，国内镜像）
    web         - 启动 Web UI（可视化调参）
    infer       - TTS 推理（输出可复现命令）
    transcribe  - 语音转文字（音频/视频 -> 文本）
    history     - 查看 TTS 历史记录
    doctor      - 诊断环境问题

用法：
    python -m tts_skill setup
    python -m tts_skill web --port 7860
    python -m tts_skill infer --text "你好" --mode auto --seed 42
    python -m tts_skill transcribe --input video.mp4
    python -m tts_skill history
    python -m tts_skill doctor
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from tts_skill.utils import (
    PROJECT_ROOT,
    detect_torch_device,
    get_platform,
    get_recommended_device_hint,
    is_china_network,
    is_setup_done,
    set_hf_mirror_env,
)


def _setup_logging(verbose: bool = False) -> None:
    """配置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    """安装环境。"""
    from tts_skill.setup_env import setup_environment

    ok, log = setup_environment(force=args.force, skip_tn=args.skip_tn, skip_asr=args.skip_asr)
    print(log)
    return 0 if ok else 1


def cmd_web(args: argparse.Namespace) -> int:
    """启动 Web UI。"""
    from tts_skill.setup_env import ensure_setup

    # 自动安装环境
    if not ensure_setup(auto_install=not args.no_auto_setup):
        print("环境未就绪，请先运行: python -m tts_skill setup")
        return 1

    from tts_skill.webui import run_webui

    return run_webui(
        model_name=args.model,
        device=args.device,
        ip=args.ip,
        port=args.port,
        share=args.share,
    )


def cmd_infer(args: argparse.Namespace) -> int:
    """CLI 推理。"""
    from tts_skill.setup_env import ensure_setup

    if not ensure_setup(auto_install=not args.no_auto_setup):
        print("环境未就绪，请先运行: python -m tts_skill setup")
        return 1

    from tts_skill.infer import run_inference

    return run_inference(args)


def cmd_transcribe(args: argparse.Namespace) -> int:
    """语音转文字。"""
    from tts_skill.setup_env import ensure_setup

    # ASR 依赖是 setup 的一部分；若未安装则尝试自动安装
    if not ensure_setup(auto_install=not args.no_auto_setup):
        print("环境未就绪，请先运行: python -m tts_skill setup")
        return 1

    from tts_skill.transcribe import run_transcribe

    return run_transcribe(args)


def cmd_history(args: argparse.Namespace) -> int:
    """查看历史记录。"""
    from tts_skill.config import build_reproducible_command, load_history

    history_type = getattr(args, "type", "all")
    has_any = False

    # TTS 生成记录
    if history_type in ("tts", "all"):
        records = load_history()
        if records:
            has_any = True
            print(f"=== TTS 生成历史（共 {len(records)} 条）===\n")
            for i, r in enumerate(records, 1):
                print(f"[{i}] ID: {r.id}")
                print(f"    名称: {r.name}")
                print(f"    模式: {r.mode}  时间: {r.timestamp}")
                print(f"    种子: {r.seed}  批量: {r.batch_count}")
                print(f"    文本: {r.text[:80]}{'...' if len(r.text) > 80 else ''}")
                if r.instruct:
                    print(f"    属性: {r.instruct}")
                if r.ref_audio:
                    print(f"    参考: {r.ref_audio}")
                print(f"    输出: {len(r.output_files)} 个文件")
                if r.output_files:
                    print(f"          {r.output_files[0]}")
                if args.show_cmd:
                    print(f"    复现命令:")
                    print(f"      {build_reproducible_command(r)}")
                print()

    # ASR 转写记录
    if history_type in ("asr", "all"):
        from tts_skill.transcribe import (
            build_transcribe_reproducible_command,
            load_transcribe_history,
        )

        asr_records = load_transcribe_history()
        if asr_records:
            has_any = True
            print(f"=== 语音转写历史（共 {len(asr_records)} 条）===\n")
            for i, r in enumerate(asr_records, 1):
                print(f"[{i}] ID: {r.id}")
                print(f"    名称: {r.name}")
                print(f"    时间: {r.timestamp}")
                print(f"    输入: {r.input_file} ({r.file_type})")
                print(f"    语言: {r.language} (置信度 {r.language_probability:.2%})")
                print(f"    时长: {r.duration:.1f}s  分段: {r.segment_count}")
                print(f"    模型: {r.model_size} (device={r.device}, task={r.task})")
                preview = r.text[:80]
                if len(r.text) > 80:
                    preview += "..."
                print(f"    文本: {preview}")
                print(f"    输出: {len(r.output_files)} 个文件")
                if r.output_files:
                    print(f"          {r.output_files[0]}")
                if args.show_cmd:
                    print(f"    复现命令:")
                    print(f"      {build_transcribe_reproducible_command(r)}")
                print()

    if not has_any:
        print("暂无历史记录")

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """诊断环境问题。"""
    print("=" * 60)
    print("OmniVoice TTS Skill 环境诊断")
    print("=" * 60)

    # 平台信息
    plat = get_platform()
    print(f"\n[平台]")
    print(f"  操作系统: {plat}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Python 路径: {sys.executable}")
    print(f"  项目根目录: {PROJECT_ROOT}")

    # 网络区域
    print(f"\n[网络]")
    print(f"  国内镜像: {'启用' if is_china_network() else '未启用'}")
    print(f"  HF_ENDPOINT: {__import__('os').environ.get('HF_ENDPOINT', '未设置')}")

    # 设备检测
    print(f"\n[设备]")
    device_hint = get_recommended_device_hint()
    print(f"  推荐设备: {device_hint}")

    # 安装状态
    print(f"\n[安装状态]")
    print(f"  setup 标记: {'已完成' if is_setup_done() else '未完成'}")

    # torch
    try:
        import torch

        print(f"  torch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: 可用 ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  CUDA: 不可用")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"  XPU: 可用")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  MPS: 可用")
        print(f"  检测到设备: {detect_torch_device()}")
    except ImportError:
        print(f"  torch: 未安装")

    # omnivoice
    try:
        import omnivoice

        print(f"  omnivoice: 已安装 ({getattr(omnivoice, '__version__', 'unknown')})")
    except ImportError:
        print(f"  omnivoice: 未安装")

    # 语音转文字（ASR）依赖
    print(f"\n[语音转文字 ASR]")
    try:
        import faster_whisper

        print(f"  faster-whisper: {getattr(faster_whisper, '__version__', 'unknown')}")
    except ImportError:
        print(f"  faster-whisper: 未安装（transcribe 命令不可用）")
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"  imageio-ffmpeg: 已安装")
        print(f"  ffmpeg 路径: {ffmpeg_path}")
    except (ImportError, Exception):
        print(f"  imageio-ffmpeg: 未安装（视频转写不可用）")

    # gradio
    try:
        import gradio

        print(f"  gradio: {gradio.__version__}")
    except ImportError:
        print(f"  gradio: 未安装")

    # soundfile
    try:
        import soundfile as sf

        print(f"  soundfile: {sf.__version__}")
    except ImportError:
        print(f"  soundfile: 未安装")

    # 诊断建议
    print(f"\n[建议]")
    if not is_setup_done():
        print("  - 运行 'python -m tts_skill setup' 安装环境")
    try:
        import torch  # noqa: F401

        import omnivoice  # noqa: F401

        print("  - TTS 环境正常，可以开始使用")
    except ImportError:
        print("  - 缺少 TTS 依赖，请运行 'python -m tts_skill setup'")
    try:
        import faster_whisper  # noqa: F401

        import imageio_ffmpeg  # noqa: F401

        print("  - ASR 环境正常，可以使用 transcribe 命令")
    except ImportError:
        print("  - 缺少 ASR 依赖，请运行 'python -m tts_skill setup'（或 --skip_asr=false）")

    print("\n" + "=" * 60)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """显示版本信息。"""
    from tts_skill import __version__

    print(f"tts-skill v{__version__}")
    print("基于 OmniVoice: https://github.com/k2-fsa/OmniVoice")
    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def build_main_parser() -> argparse.ArgumentParser:
    """构建主参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="tts_skill",
        description="OmniVoice TTS Skill - 一键式声音克隆与生成工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志输出")
    parser.add_argument("--version", action="store_true", help="显示版本信息")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # setup
    p_setup = subparsers.add_parser("setup", help="安装环境（跨平台，国内镜像）")
    p_setup.add_argument("--force", action="store_true", help="强制重新安装")
    p_setup.add_argument("--skip_tn", action="store_true", help="跳过文本归一化依赖")
    p_setup.add_argument("--skip_asr", action="store_true", help="跳过语音转文字（ASR）依赖")
    p_setup.set_defaults(func=cmd_setup)

    # web
    p_web = subparsers.add_parser("web", help="启动 Web UI")
    from tts_skill.webui import build_webui_parser

    web_parser = build_webui_parser()
    for action in web_parser._actions:
        if action.dest in ("help",):
            continue
        p_web._add_action(action)
    p_web.add_argument("--no_auto_setup", action="store_true", help="不自动安装环境")
    p_web.set_defaults(func=cmd_web)

    # infer
    p_infer = subparsers.add_parser("infer", help="CLI 推理")
    from tts_skill.infer import build_infer_parser

    infer_parser = build_infer_parser()
    for action in infer_parser._actions:
        if action.dest in ("help",):
            continue
        p_infer._add_action(action)
    p_infer.add_argument("--no_auto_setup", action="store_true", help="不自动安装环境")
    p_infer.set_defaults(func=cmd_infer)

    # transcribe
    p_transcribe = subparsers.add_parser(
        "transcribe", help="语音转文字（音频/视频 -> 文本）"
    )
    from tts_skill.transcribe import build_transcribe_parser

    transcribe_parser = build_transcribe_parser()
    for action in transcribe_parser._actions:
        if action.dest in ("help",):
            continue
        p_transcribe._add_action(action)
    p_transcribe.add_argument("--no_auto_setup", action="store_true", help="不自动安装环境")
    p_transcribe.set_defaults(func=cmd_transcribe)

    # history
    p_history = subparsers.add_parser("history", help="查看历史记录")
    p_history.add_argument("--show_cmd", action="store_true", help="显示复现命令")
    p_history.add_argument(
        "--type",
        choices=["tts", "asr", "all"],
        default="all",
        help="历史记录类型：tts=生成记录, asr=转写记录, all=全部（默认 all）",
    )
    p_history.set_defaults(func=cmd_history)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="诊断环境问题")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """主入口函数。"""
    parser = build_main_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if not args.command:
        parser.print_help()
        return 0

    _setup_logging(getattr(args, "verbose", False))
    set_hf_mirror_env()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as e:
        logging.error("执行失败: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
