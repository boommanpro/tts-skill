"""CLI 推理模块：单次/批量生成音频，输出可复现命令。

支持三种模式：
- clone: 声音克隆（需 ref_audio + ref_text）
- design: 声音设计（需 instruct）
- auto: 自动声音（无参考）

用法：
    python -m tts_skill infer --text "你好" --mode auto --seed 42
    python -m tts_skill infer --text "Hello" --ref_audio ref.wav --mode clone
    python -m tts_skill infer --text "Hello" --instruct "male, british accent" --mode design --batch_count 5
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from tts_skill.config import (
    GenerationRecord,
    build_reproducible_command,
    create_record,
)
from tts_skill.utils import (
    DEFAULT_MODEL,
    OUTPUT_DIR,
    detect_torch_device,
    ensure_output_dir,
    fix_random_seed,
    gen_random_seed,
    get_platform,
    is_setup_done,
    set_hf_mirror_env,
)

logger = logging.getLogger(__name__)


def _str2bool(v: str) -> bool:
    """argparse bool 类型解析。"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"无效的布尔值: {v}")


def build_infer_parser() -> argparse.ArgumentParser:
    """构建推理命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="tts_skill infer",
        description="OmniVoice 声音生成 / 克隆（支持随机种子、批量生成、可复现命令）",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 必需参数
    parser.add_argument("--text", type=str, required=True, help="要合成的文本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clone", "design", "auto"],
        default="auto",
        help="生成模式：clone=声音克隆, design=声音设计, auto=自动（默认 auto）",
    )

    # 声音克隆
    parser.add_argument("--ref_audio", type=str, default=None, help="参考音频路径（clone 模式必需）")
    parser.add_argument("--ref_text", type=str, default=None, help="参考音频文本（可选，留空自动转写）")

    # 声音设计
    parser.add_argument("--instruct", type=str, default=None, help="声音属性描述（design 模式，如 'male, british accent'）")

    # 语种
    parser.add_argument("--language", type=str, default=None, help="语种名称或代码（如 'Chinese', 'en'），可选")

    # 随机种子
    parser.add_argument("--seed", type=int, default=None, help="随机种子（相同种子可复现结果，默认随机）")

    # 生成参数
    parser.add_argument("--num_step", type=int, default=32, help="扩散解码步数（默认 32，更小更快）")
    parser.add_argument("--speed", type=float, default=1.0, help="语速（>1 更快，<1 更慢）")
    parser.add_argument("--duration", type=float, default=None, help="固定输出时长（秒），设置后覆盖 speed")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="分类器自由引导尺度（默认 2.0）")
    parser.add_argument("--t_shift", type=float, default=0.1, help="时间步偏移（默认 0.1）")
    parser.add_argument("--denoise", type=_str2bool, default=True, help="是否启用降噪 token（默认 true）")
    parser.add_argument("--postprocess_output", type=_str2bool, default=True, help="是否后处理输出（默认 true）")
    parser.add_argument("--normalize_text", type=_str2bool, default=False, help="是否文本归一化（默认 false）")

    # 批量生成
    parser.add_argument("--batch_count", type=int, default=1, help="批量生成数量（默认 1，每个使用不同种子）")

    # 输出
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--name", type=str, default=None, help="生成记录名称（用于历史记录）")

    # 模型与设备
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称或路径")
    parser.add_argument("--device", type=str, default=None, help="推理设备（自动检测）")

    return parser


def _load_model(model_name: str, device: Optional[str] = None):
    """加载 OmniVoice 模型。"""
    import torch

    from omnivoice import OmniVoice

    if device is None:
        device = detect_torch_device()

    logger.info("加载模型 %s (device=%s)...", model_name, device)
    model = OmniVoice.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float16,
    )
    logger.info("模型加载完成，采样率: %d Hz", model.sampling_rate)
    return model


def _generate_one(
    model,
    *,
    text: str,
    mode: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
    instruct: Optional[str],
    language: Optional[str],
    seed: int,
    num_step: int,
    speed: float,
    duration: Optional[float],
    guidance_scale: float,
    t_shift: float,
    denoise: bool,
    postprocess_output: bool,
    normalize_text: bool,
):
    """执行单次生成，返回 numpy 音频数组。"""
    fix_random_seed(seed)

    kwargs = {
        "text": text,
        "language": language,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "t_shift": t_shift,
        "denoise": denoise,
        "postprocess_output": postprocess_output,
        "normalize_text": normalize_text,
    }
    if speed != 1.0:
        kwargs["speed"] = speed
    if duration is not None and duration > 0:
        kwargs["duration"] = duration

    if mode == "clone":
        if not ref_audio:
            raise ValueError("clone 模式需要 --ref_audio")
        kwargs["ref_audio"] = ref_audio
        kwargs["ref_text"] = ref_text
    elif mode == "design":
        if not instruct:
            raise ValueError("design 模式需要 --instruct")
        kwargs["instruct"] = instruct
    # auto 模式无额外参数

    audio = model.generate(**kwargs)
    return audio[0]


def run_inference(args: argparse.Namespace) -> int:
    """执行推理流程，返回退出码。"""
    set_hf_mirror_env()
    ensure_output_dir()

    # 确定种子
    base_seed = args.seed if args.seed is not None else gen_random_seed()
    batch_seeds = [base_seed + i for i in range(args.batch_count)]

    # 验证参数
    if args.mode == "clone" and not args.ref_audio:
        logger.error("clone 模式需要 --ref_audio 参数")
        return 1
    if args.mode == "design" and not args.instruct:
        logger.error("design 模式需要 --instruct 参数")
        return 1

    # 检查环境
    if not is_setup_done():
        logger.error("环境未安装，请先运行: python -m tts_skill setup")
        return 1

    # 加载模型
    try:
        model = _load_model(args.model, args.device)
    except Exception as e:
        logger.error("模型加载失败: %s", e)
        return 1

    # 批量生成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    device_used = args.device or detect_torch_device()

    import soundfile as sf

    for i, seed in enumerate(batch_seeds):
        logger.info(
            "[%d/%d] 生成中 (seed=%d, mode=%s)...",
            i + 1,
            len(batch_seeds),
            seed,
            args.mode,
        )
        try:
            audio = _generate_one(
                model,
                text=args.text,
                mode=args.mode,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                instruct=args.instruct,
                language=args.language,
                seed=seed,
                num_step=args.num_step,
                speed=args.speed,
                duration=args.duration,
                guidance_scale=args.guidance_scale,
                t_shift=args.t_shift,
                denoise=args.denoise,
                postprocess_output=args.postprocess_output,
                normalize_text=args.normalize_text,
            )
        except Exception as e:
            logger.error("[%d/%d] 生成失败: %s", i + 1, len(batch_seeds), e)
            return 1

        # 保存文件
        suffix = f"_{i}" if args.batch_count > 1 else ""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{args.mode}_{timestamp}{suffix}_seed{seed}.wav"
        output_path = output_dir / filename
        sf.write(str(output_path), audio, model.sampling_rate)
        output_files.append(str(output_path))
        logger.info("[%d/%d] 已保存: %s", i + 1, len(batch_seeds), output_path)

    # 创建历史记录
    record = create_record(
        mode=args.mode,
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        language=args.language,
        seed=base_seed,
        num_step=args.num_step,
        speed=args.speed,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        normalize_text=args.normalize_text,
        batch_count=args.batch_count,
        batch_seeds=batch_seeds,
        output_files=output_files,
        device=device_used,
        name=args.name,
    )

    # 输出可复现命令
    reproducible_cmd = build_reproducible_command(record)
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"记录 ID: {record.id}")
    print(f"记录名称: {record.name}")
    print(f"基础种子: {base_seed}")
    print(f"批量种子: {batch_seeds}")
    print(f"输出文件:")
    for f in output_files:
        print(f"  - {f}")
    print()
    print("可复现命令（复制以下命令可重新生成相同结果）:")
    print("-" * 60)
    print(reproducible_cmd)
    print("-" * 60)

    return 0
