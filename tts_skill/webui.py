"""Gradio Web UI：可视化调参、批量生成、历史记录管理。

功能：
- 三种生成模式（声音克隆 / 声音设计 / 自动声音）
- 可视化随机种子配置（固定种子 / 随机种子）
- 批量生成（默认 5 个，每个不同种子）
- 历史记录查看、名称编辑、删除
- 生成后输出可复现 CLI 命令
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

from tts_skill.config import (
    GenerationRecord,
    build_reproducible_command,
    create_record,
    delete_record,
    load_history,
    update_record_name,
)
from tts_skill.utils import (
    DEFAULT_MODEL,
    detect_torch_device,
    ensure_output_dir,
    fix_random_seed,
    gen_random_seed,
    is_setup_done,
    set_hf_mirror_env,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 声音设计属性选项（对齐 omnivoice/cli/demo.py）
# ---------------------------------------------------------------------------

_DESIGN_CATEGORIES = {
    "性别 / Gender": ["Auto", "男 / Male", "女 / Female"],
    "年龄 / Age": [
        "Auto",
        "儿童 / Child",
        "少年 / Teenager",
        "青年 / Young Adult",
        "中年 / Middle-aged",
        "老年 / Elderly",
    ],
    "音调 / Pitch": [
        "Auto",
        "极低音调 / Very Low Pitch",
        "低音调 / Low Pitch",
        "中音调 / Moderate Pitch",
        "高音调 / High Pitch",
        "极高音调 / Very High Pitch",
    ],
    "风格 / Style": ["Auto", "耳语 / Whisper"],
    "英文口音 / English Accent": [
        "Auto",
        "美式口音 / American Accent",
        "英式口音 / British Accent",
        "澳大利亚口音 / Australian Accent",
        "加拿大口音 / Canadian Accent",
        "印度口音 / Indian Accent",
        "中国口音 / Chinese Accent",
        "日本口音 / Japanese Accent",
        "韩国口音 / Korean Accent",
        "葡萄牙口音 / Portuguese Accent",
        "俄罗斯口音 / Russian Accent",
    ],
    "中文方言 / Chinese Dialect": [
        "Auto",
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    ],
}


def _build_instruct_from_selections(selections: list[str]) -> Optional[str]:
    """从下拉选择构建 instruct 字符串。

    选项格式为 "中文 / English"：
    - 普通属性取英文部分（如 "男 / Male" -> "Male"）
    - 方言类取中文部分（如 "四川话" 无 " / "，直接保留）
    """
    parts = []
    for sel in selections:
        if not sel or sel == "Auto":
            continue
        if " / " in sel:
            # 格式为 "中文 / English"，取英文部分（第二段）
            zh, en = sel.split(" / ", 1)
            # 方言类（中文部分含"话"）使用中文
            if "话" in zh:
                parts.append(zh.strip())
            else:
                parts.append(en.strip())
        else:
            # 无分隔符的选项（如纯方言"四川话"）直接使用
            parts.append(sel.strip())
    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str, device: Optional[str] = None):
    """加载并缓存模型。"""
    cache_key = f"{model_name}|{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

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
    _MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# 生成核心逻辑
# ---------------------------------------------------------------------------


def _generate_audio(
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
    """执行单次生成，返回 (audio_np, sample_rate)。"""
    fix_random_seed(seed)

    kwargs = {
        "text": text.strip(),
        "language": language if language else None,
        "num_step": int(num_step),
        "guidance_scale": float(guidance_scale),
        "t_shift": float(t_shift),
        "denoise": bool(denoise),
        "postprocess_output": bool(postprocess_output),
        "normalize_text": bool(normalize_text),
    }
    if speed is not None and float(speed) != 1.0:
        kwargs["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kwargs["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            raise ValueError("请上传参考音频")
        kwargs["ref_audio"] = ref_audio
        kwargs["ref_text"] = ref_text if ref_text else None
    elif mode == "design":
        if not instruct:
            raise ValueError("请选择至少一个声音属性")
        kwargs["instruct"] = instruct

    audio = model.generate(**kwargs)
    return audio[0], model.sampling_rate


def _do_batch_generate(
    model,
    *,
    text: str,
    mode: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
    instruct: Optional[str],
    language: Optional[str],
    base_seed: int,
    batch_count: int,
    num_step: int,
    speed: float,
    duration: Optional[float],
    guidance_scale: float,
    t_shift: float,
    denoise: bool,
    postprocess_output: bool,
    normalize_text: bool,
    output_dir: Path,
    name: Optional[str] = None,
):
    """批量生成并保存，返回 (record, output_files, audio_samples)。"""
    import numpy as np
    import soundfile as sf

    batch_seeds = [base_seed + i for i in range(batch_count)]
    output_files: list[str] = []
    audio_samples: list[tuple[int, np.ndarray]] = []
    errors: list[str] = []

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for i, seed in enumerate(batch_seeds):
        try:
            audio, sr = _generate_audio(
                model,
                text=text,
                mode=mode,
                ref_audio=ref_audio,
                ref_text=ref_text,
                instruct=instruct,
                language=language,
                seed=seed,
                num_step=num_step,
                speed=speed,
                duration=duration,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                denoise=denoise,
                postprocess_output=postprocess_output,
                normalize_text=normalize_text,
            )
            suffix = f"_{i}" if batch_count > 1 else ""
            filename = f"{mode}_{timestamp}{suffix}_seed{seed}.wav"
            output_path = output_dir / filename
            sf.write(str(output_path), audio, sr)
            output_files.append(str(output_path))
            audio_samples.append((sr, audio))
        except Exception as e:
            errors.append(f"[seed={seed}] {type(e).__name__}: {e}")
            logger.exception("生成失败 seed=%d", seed)

    if not output_files:
        raise RuntimeError("所有生成均失败: " + "; ".join(errors))

    # 创建历史记录
    record = create_record(
        mode=mode,
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct=instruct,
        language=language,
        seed=base_seed,
        num_step=num_step,
        speed=speed,
        duration=duration,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        denoise=denoise,
        postprocess_output=postprocess_output,
        normalize_text=normalize_text,
        batch_count=batch_count,
        batch_seeds=batch_seeds,
        output_files=output_files,
        device=str(model.device),
        name=name,
    )

    return record, output_files, audio_samples, errors


# ---------------------------------------------------------------------------
# 历史记录表格数据
# ---------------------------------------------------------------------------


def _history_table_data() -> list[list[str]]:
    """获取历史记录表格数据（list 格式，供 Dataframe 使用）。"""
    records = load_history()
    rows = []
    for r in records:
        rows.append(
            [
                r.id,
                r.name,
                r.mode,
                r.timestamp,
                str(r.seed),
                str(r.batch_count),
                ", ".join(Path(f).name for f in r.output_files) if r.output_files else "",
            ]
        )
    return rows


def _history_markdown() -> str:
    """获取历史记录的 Markdown 表格（不依赖 pandas/matplotlib，更健壮）。"""
    records = load_history()
    if not records:
        return "_暂无历史记录_"

    lines = [
        "| ID | 名称 | 模式 | 时间 | 种子 | 批量 | 输出文件 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        files = ", ".join(Path(f).name for f in r.output_files) if r.output_files else "-"
        # 转义 Markdown 中的特殊字符
        name = r.name.replace("|", "\\|")
        lines.append(
            f"| `{r.id}` | {name} | {r.mode} | {r.timestamp} | {r.seed} | {r.batch_count} | {files} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 构建 Gradio 界面
# ---------------------------------------------------------------------------


def build_demo(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> Any:
    """构建 Gradio Web UI。"""
    import gradio as gr
    import numpy as np

    output_dir = ensure_output_dir()

    # 批量输出音频组件的最大数量（与 UI 中预创建的 gr.Audio 数量一致）
    MAX_BATCH_AUDIO = 5

    def _pad_audio_outputs(audio_samples, errors):
        """将音频样本列表填充/截断为固定长度 MAX_BATCH_AUDIO。

        每个元素为 (sr, audio_int16) 或 None（无该序号结果）。
        """
        outputs: list = []
        for i in range(MAX_BATCH_AUDIO):
            if i < len(audio_samples):
                sr, audio = audio_samples[i]
                audio_int16 = (audio * 32767).astype(np.int16)
                outputs.append((sr, audio_int16))
            else:
                outputs.append(None)
        return outputs

    def _build_status(output_files, base_seed, errors, extra: str = ""):
        """构建状态文本。"""
        status = f"生成完成！共 {len(output_files)} 个文件，基础种子: {base_seed}"
        if errors:
            status += f"\n部分失败: {len(errors)} 个"
        if extra:
            status += f"\n{extra}"
        return status

    # -- 生成函数（供 UI 调用）--
    def _generate_clone(
        text, ref_audio, ref_text, language,
        seed_input, use_random_seed, batch_count,
        num_step, speed, duration, guidance_scale, t_shift,
        denoise, postprocess_output, normalize_text,
        name,
    ):
        empty_audio = tuple([None] * MAX_BATCH_AUDIO)
        if not text or not text.strip():
            return *empty_audio, "请输入要合成的文本", "", _history_markdown()
        if not ref_audio:
            return *empty_audio, "请上传参考音频", "", _history_markdown()

        try:
            model = _get_model(model_name, device)
        except Exception as e:
            return *empty_audio, f"模型加载失败: {e}", "", _history_markdown()

        base_seed = gen_random_seed() if use_random_seed else int(seed_input)

        try:
            record, output_files, audio_samples, errors = _do_batch_generate(
                model,
                text=text,
                mode="clone",
                ref_audio=ref_audio,
                ref_text=ref_text,
                instruct=None,
                language=language if language != "Auto" else None,
                base_seed=base_seed,
                batch_count=int(batch_count),
                num_step=num_step,
                speed=speed,
                duration=duration if duration else None,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                denoise=denoise,
                postprocess_output=postprocess_output,
                normalize_text=normalize_text,
                output_dir=output_dir,
                name=name if name else None,
            )
        except Exception as e:
            return *empty_audio, f"生成失败: {e}", "", _history_markdown()

        audio_outputs = _pad_audio_outputs(audio_samples, errors)
        status = _build_status(output_files, base_seed, errors)
        cmd = build_reproducible_command(record)
        return *audio_outputs, status, cmd, _history_markdown()

    def _generate_design(
        text, language,
        gender, age, pitch, style, accent, dialect,
        seed_input, use_random_seed, batch_count,
        num_step, speed, duration, guidance_scale, t_shift,
        denoise, postprocess_output, normalize_text,
        name,
    ):
        empty_audio = tuple([None] * MAX_BATCH_AUDIO)
        if not text or not text.strip():
            return *empty_audio, "请输入要合成的文本", "", _history_markdown()

        selections = [gender, age, pitch, style, accent, dialect]
        instruct = _build_instruct_from_selections(selections)
        if not instruct:
            return *empty_audio, "请至少选择一个声音属性", "", _history_markdown()

        try:
            model = _get_model(model_name, device)
        except Exception as e:
            return *empty_audio, f"模型加载失败: {e}", "", _history_markdown()

        base_seed = gen_random_seed() if use_random_seed else int(seed_input)

        try:
            record, output_files, audio_samples, errors = _do_batch_generate(
                model,
                text=text,
                mode="design",
                ref_audio=None,
                ref_text=None,
                instruct=instruct,
                language=language if language != "Auto" else None,
                base_seed=base_seed,
                batch_count=int(batch_count),
                num_step=num_step,
                speed=speed,
                duration=duration if duration else None,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                denoise=denoise,
                postprocess_output=postprocess_output,
                normalize_text=normalize_text,
                output_dir=output_dir,
                name=name if name else None,
            )
        except Exception as e:
            return *empty_audio, f"生成失败: {e}", "", _history_markdown()

        audio_outputs = _pad_audio_outputs(audio_samples, errors)
        status = _build_status(output_files, base_seed, errors, f"声音属性: {instruct}")
        cmd = build_reproducible_command(record)
        return *audio_outputs, status, cmd, _history_markdown()

    def _generate_auto(
        text, language,
        seed_input, use_random_seed, batch_count,
        num_step, speed, duration, guidance_scale, t_shift,
        denoise, postprocess_output, normalize_text,
        name,
    ):
        empty_audio = tuple([None] * MAX_BATCH_AUDIO)
        if not text or not text.strip():
            return *empty_audio, "请输入要合成的文本", "", _history_markdown()

        try:
            model = _get_model(model_name, device)
        except Exception as e:
            return *empty_audio, f"模型加载失败: {e}", "", _history_markdown()

        base_seed = gen_random_seed() if use_random_seed else int(seed_input)

        try:
            record, output_files, audio_samples, errors = _do_batch_generate(
                model,
                text=text,
                mode="auto",
                ref_audio=None,
                ref_text=None,
                instruct=None,
                language=language if language != "Auto" else None,
                base_seed=base_seed,
                batch_count=int(batch_count),
                num_step=num_step,
                speed=speed,
                duration=duration if duration else None,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                denoise=denoise,
                postprocess_output=postprocess_output,
                normalize_text=normalize_text,
                output_dir=output_dir,
                name=name if name else None,
            )
        except Exception as e:
            return *empty_audio, f"生成失败: {e}", "", _history_markdown()

        audio_outputs = _pad_audio_outputs(audio_samples, errors)
        status = _build_status(output_files, base_seed, errors)
        cmd = build_reproducible_command(record)
        return *audio_outputs, status, cmd, _history_markdown()

    # -- 历史记录操作 --
    def _rename_record(record_id, new_name):
        if not record_id:
            return "请输入要重命名的记录 ID", _history_markdown()
        if update_record_name(record_id, new_name):
            return f"已重命名 {record_id} -> {new_name}", _history_markdown()
        return f"未找到记录: {record_id}", _history_markdown()

    def _delete_record(record_id):
        if not record_id:
            return "请输入要删除的记录 ID", _history_markdown()
        if delete_record(record_id):
            return f"已删除记录: {record_id}", _history_markdown()
        return f"未找到记录: {record_id}", _history_markdown()

    def _show_reproducible_cmd(record_id):
        records = load_history()
        for r in records:
            if r.id == record_id:
                return build_reproducible_command(r)
        return f"未找到记录: {record_id}"

    # =====================================================================
    # UI 布局
    # =====================================================================

    theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
    css = """
    .gradio-container {max-width: 100% !important; font-size: 15px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    """

    with gr.Blocks(theme=theme, css=css, title="OmniVoice TTS Skill") as demo:
        gr.Markdown(
            """
            # OmniVoice TTS Skill

            基于开源模型实现的声音克隆与生成工具，支持：
            - 声音克隆（参考音频）、声音设计（属性描述）、自动声音
            - 可视化随机种子配置（相同种子可复现）
            - 批量生成（默认 5 个）、历史记录、名称编辑
            - 生成后输出可复现 CLI 命令
            """
        )

        # --- 通用：种子与批量生成设置组件 ---
        def _seed_batch_components():
            with gr.Accordion("随机种子与批量生成", open=True):
                with gr.Row():
                    seed_input = gr.Number(
                        label="随机种子",
                        value=42,
                        info="相同种子 + 相同参数 = 可复现结果",
                    )
                    use_random_seed = gr.Checkbox(
                        label="使用随机种子（忽略上方种子值）",
                        value=False,
                    )
                    batch_count = gr.Number(
                        label="批量生成数量",
                        value=5,
                        precision=0,
                        minimum=1,
                        maximum=20,
                        info="每个使用不同种子（base_seed + i）",
                    )
            return seed_input, use_random_seed, batch_count

        def _gen_settings_components():
            with gr.Accordion("生成参数（高级）", open=False):
                with gr.Row():
                    num_step = gr.Slider(4, 64, value=32, step=1, label="解码步数 num_step", info="更小更快，更大质量更好")
                    guidance_scale = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="引导尺度 guidance_scale")
                    t_shift = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="时间步偏移 t_shift")
                with gr.Row():
                    speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="语速 speed", info=">1 更快，<1 更慢")
                    duration = gr.Number(label="固定时长(秒) duration", value=None, info="设置后覆盖 speed")
                with gr.Row():
                    denoise = gr.Checkbox(label="降噪 denoise", value=True)
                    postprocess_output = gr.Checkbox(label="后处理 postprocess_output", value=True)
                    normalize_text = gr.Checkbox(label="文本归一化 normalize_text", value=False)
            return num_step, guidance_scale, t_shift, speed, duration, denoise, postprocess_output, normalize_text

        # 全局共享的历史记录 Markdown 组件
        # 定义在 Tabs 之前，所有 Tab 的生成函数都会更新它
        gr.Markdown("## 最近生成历史")
        history_md_state = gr.Markdown(value=_history_markdown())

        with gr.Tabs():
            # =============================================================
            # Tab 1: 声音克隆
            # =============================================================
            with gr.TabItem("声音克隆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(label="要合成的文本", lines=4, placeholder="输入要合成的文字...")
                        vc_ref_audio = gr.Audio(label="参考音频（3-10 秒）", type="filepath")
                        vc_ref_text = gr.Textbox(
                            label="参考音频文本（可选）",
                            lines=2,
                            placeholder="留空则自动转写",
                        )
                        vc_lang = gr.Textbox(
                            label="语种（可选，如 Chinese, en）",
                            value="",
                            placeholder="留空自动检测",
                        )
                        vc_name = gr.Textbox(label="记录名称（可选）", value="", placeholder="如：我的声音克隆1")
                        vc_seed, vc_use_rand, vc_batch = _seed_batch_components()
                        vc_ns, vc_gs, vc_ts, vc_sp, vc_du, vc_dn, vc_po, vc_nt = _gen_settings_components()
                        vc_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio_outs = [
                            gr.Audio(label=f"输出音频 #{i+1}", type="numpy", visible=(i == 0))
                            for i in range(MAX_BATCH_AUDIO)
                        ]
                        vc_status = gr.Textbox(label="状态", lines=3)
                        vc_cmd = gr.Textbox(label="可复现 CLI 命令", lines=3, interactive=False)

                vc_btn.click(
                    _generate_clone,
                    inputs=[
                        vc_text, vc_ref_audio, vc_ref_text, vc_lang,
                        vc_seed, vc_use_rand, vc_batch,
                        vc_ns, vc_sp, vc_du, vc_gs, vc_ts,
                        vc_dn, vc_po, vc_nt, vc_name,
                    ],
                    outputs=[*vc_audio_outs, vc_status, vc_cmd, history_md_state],
                )

            # =============================================================
            # Tab 2: 声音设计
            # =============================================================
            with gr.TabItem("声音设计"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(label="要合成的文本", lines=4, placeholder="输入要合成的文字...")
                        vd_lang = gr.Textbox(label="语种（可选）", value="", placeholder="留空自动检测")
                        with gr.Row():
                            vd_gender = gr.Dropdown(_DESIGN_CATEGORIES["性别 / Gender"], value="Auto", label="性别")
                            vd_age = gr.Dropdown(_DESIGN_CATEGORIES["年龄 / Age"], value="Auto", label="年龄")
                        with gr.Row():
                            vd_pitch = gr.Dropdown(_DESIGN_CATEGORIES["音调 / Pitch"], value="Auto", label="音调")
                            vd_style = gr.Dropdown(_DESIGN_CATEGORIES["风格 / Style"], value="Auto", label="风格")
                        with gr.Row():
                            vd_accent = gr.Dropdown(_DESIGN_CATEGORIES["英文口音 / English Accent"], value="Auto", label="英文口音")
                            vd_dialect = gr.Dropdown(_DESIGN_CATEGORIES["中文方言 / Chinese Dialect"], value="Auto", label="中文方言")
                        vd_name = gr.Textbox(label="记录名称（可选）", value="")
                        vd_seed, vd_use_rand, vd_batch = _seed_batch_components()
                        vd_ns, vd_gs, vd_ts, vd_sp, vd_du, vd_dn, vd_po, vd_nt = _gen_settings_components()
                        vd_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio_outs = [
                            gr.Audio(label=f"输出音频 #{i+1}", type="numpy", visible=(i == 0))
                            for i in range(MAX_BATCH_AUDIO)
                        ]
                        vd_status = gr.Textbox(label="状态", lines=3)
                        vd_cmd = gr.Textbox(label="可复现 CLI 命令", lines=3, interactive=False)

                vd_btn.click(
                    _generate_design,
                    inputs=[
                        vd_text, vd_lang,
                        vd_gender, vd_age, vd_pitch, vd_style, vd_accent, vd_dialect,
                        vd_seed, vd_use_rand, vd_batch,
                        vd_ns, vd_sp, vd_du, vd_gs, vd_ts,
                        vd_dn, vd_po, vd_nt, vd_name,
                    ],
                    outputs=[*vd_audio_outs, vd_status, vd_cmd, history_md_state],
                )

            # =============================================================
            # Tab 3: 自动声音
            # =============================================================
            with gr.TabItem("自动声音"):
                with gr.Row():
                    with gr.Column(scale=1):
                        va_text = gr.Textbox(label="要合成的文本", lines=4, placeholder="输入要合成的文字...")
                        va_lang = gr.Textbox(label="语种（可选）", value="", placeholder="留空自动检测")
                        va_name = gr.Textbox(label="记录名称（可选）", value="")
                        va_seed, va_use_rand, va_batch = _seed_batch_components()
                        va_ns, va_gs, va_ts, va_sp, va_du, va_dn, va_po, va_nt = _gen_settings_components()
                        va_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        va_audio_outs = [
                            gr.Audio(label=f"输出音频 #{i+1}", type="numpy", visible=(i == 0))
                            for i in range(MAX_BATCH_AUDIO)
                        ]
                        va_status = gr.Textbox(label="状态", lines=3)
                        va_cmd = gr.Textbox(label="可复现 CLI 命令", lines=3, interactive=False)

                va_btn.click(
                    _generate_auto,
                    inputs=[
                        va_text, va_lang,
                        va_seed, va_use_rand, va_batch,
                        va_ns, va_sp, va_du, va_gs, va_ts,
                        va_dn, va_po, va_nt, va_name,
                    ],
                    outputs=[*va_audio_outs, va_status, va_cmd, history_md_state],
                )

            # =============================================================
            # Tab 4: 历史记录
            # =============================================================
            with gr.TabItem("历史记录"):
                gr.Markdown(
                    "历史记录显示在页面顶部的「最近生成历史」区域。\n"
                    "下方可对记录进行重命名、删除、查看复现命令操作。"
                )
                with gr.Row():
                    refresh_btn = gr.Button("刷新历史记录", variant="secondary")
                    refresh_btn.click(lambda: _history_markdown(), outputs=[history_md_state])

                gr.Markdown("### 重命名 / 删除 / 查看复现命令")
                with gr.Row():
                    op_record_id = gr.Textbox(label="记录 ID", placeholder="从上方历史记录区域复制 ID")
                    op_new_name = gr.Textbox(label="新名称（重命名用）", value="")
                with gr.Row():
                    rename_btn = gr.Button("重命名")
                    delete_btn = gr.Button("删除", variant="stop")
                    show_cmd_btn = gr.Button("查看复现命令")
                op_result = gr.Textbox(label="操作结果", lines=2)
                op_cmd = gr.Textbox(label="复现命令", lines=3, interactive=False)

                rename_btn.click(_rename_record, inputs=[op_record_id, op_new_name], outputs=[op_result, history_md_state])
                delete_btn.click(_delete_record, inputs=[op_record_id], outputs=[op_result, history_md_state])
                show_cmd_btn.click(_show_reproducible_cmd, inputs=[op_record_id], outputs=[op_cmd])

    return demo


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_webui(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    ip: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> int:
    """启动 Web UI。"""
    set_hf_mirror_env()

    if not is_setup_done():
        logger.error("环境未安装，请先运行: python -m tts_skill setup")
        return 1

    # 预加载模型（可选，失败也不阻塞 UI 启动）
    try:
        logger.info("预加载模型 %s...", model_name)
        _get_model(model_name, device)
    except Exception as e:
        logger.warning("模型预加载失败（UI 仍会启动，生成时会再次尝试）: %s", e)

    demo = build_demo(model_name, device)
    demo.queue().launch(server_name=ip, server_port=port, share=share)
    return 0


def build_webui_parser() -> argparse.ArgumentParser:
    """构建 Web UI 命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="tts_skill web",
        description="启动 OmniVoice TTS Skill Web UI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称或路径")
    parser.add_argument("--device", type=str, default=None, help="推理设备（自动检测）")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="监听 IP")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    return parser
