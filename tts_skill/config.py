"""配置与历史记录管理。

历史记录存储在项目根目录的 history.json，记录每次生成的完整参数，
支持名称编辑、查看、CLI 复现。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from tts_skill.utils import HISTORY_FILE, OUTPUT_DIR, ensure_output_dir


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class GenerationRecord:
    """单次生成记录（一个 record 可能包含批量生成的多个音频）。"""

    id: str
    name: str
    timestamp: str
    mode: str  # "clone" | "design" | "auto"
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    # 生成参数
    seed: int = 0
    num_step: int = 32
    speed: float = 1.0
    duration: Optional[float] = None
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    denoise: bool = True
    postprocess_output: bool = True
    normalize_text: bool = False
    # 批量生成
    batch_count: int = 1
    batch_seeds: list[int] = field(default_factory=list)
    # 输出
    output_files: list[str] = field(default_factory=list)
    # 设备
    device: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRecord":
        # 兼容缺失字段
        valid_keys = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# 历史记录存储
# ---------------------------------------------------------------------------


def _load_history_raw() -> list[dict[str, Any]]:
    """加载原始历史记录 JSON。"""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "generations" in data:
            return data["generations"]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def load_history() -> list[GenerationRecord]:
    """加载所有历史记录（按时间倒序）。"""
    raw = _load_history_raw()
    records = [GenerationRecord.from_dict(item) for item in raw]
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records


def save_history(records: list[GenerationRecord]) -> None:
    """保存历史记录（按时间正序存储）。"""
    sorted_records = sorted(records, key=lambda r: r.timestamp)
    data = [r.to_dict() for r in sorted_records]
    HISTORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_record(record: GenerationRecord) -> None:
    """添加一条记录到历史记录。"""
    records = load_history()
    records.append(record)
    save_history(records)


def get_record(record_id: str) -> Optional[GenerationRecord]:
    """根据 ID 获取记录。"""
    for r in load_history():
        if r.id == record_id:
            return r
    return None


def update_record_name(record_id: str, new_name: str) -> bool:
    """更新记录名称，返回是否成功。"""
    records = load_history()
    for r in records:
        if r.id == record_id:
            r.name = new_name
            save_history(records)
            return True
    return False


def delete_record(record_id: str) -> bool:
    """删除一条记录（同时删除输出文件），返回是否成功。"""
    records = load_history()
    remaining = []
    deleted = False
    for r in records:
        if r.id == record_id:
            deleted = True
            # 删除输出文件
            for f in r.output_files:
                try:
                    Path(f).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            remaining.append(r)
    if deleted:
        save_history(remaining)
    return deleted


def create_record(
    mode: str,
    text: str,
    *,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    seed: int = 0,
    num_step: int = 32,
    speed: float = 1.0,
    duration: Optional[float] = None,
    guidance_scale: float = 2.0,
    t_shift: float = 0.1,
    denoise: bool = True,
    postprocess_output: bool = True,
    normalize_text: bool = False,
    batch_count: int = 1,
    batch_seeds: Optional[list[int]] = None,
    output_files: Optional[list[str]] = None,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> GenerationRecord:
    """创建并保存一条生成记录。"""
    ensure_output_dir()
    record = GenerationRecord(
        id=str(uuid.uuid4())[:8],
        name=name or f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        mode=mode,
        text=text,
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
        batch_count=batch_count,
        batch_seeds=batch_seeds or [seed],
        output_files=output_files or [],
        device=device,
    )
    add_record(record)
    return record


# ---------------------------------------------------------------------------
# CLI 复现命令生成
# ---------------------------------------------------------------------------


def build_reproducible_command(record: GenerationRecord) -> str:
    """根据记录生成可复现的 CLI 命令字符串。"""
    from tts_skill.utils import get_python_executable

    parts = [get_python_executable(), "-m", "tts_skill", "infer"]
    parts += ["--text", _shell_quote(record.text)]
    parts += ["--mode", record.mode]
    parts += ["--seed", str(record.seed)]
    parts += ["--num_step", str(record.num_step)]
    parts += ["--speed", str(record.speed)]
    parts += ["--guidance_scale", str(record.guidance_scale)]
    parts += ["--t_shift", str(record.t_shift)]

    if record.duration is not None:
        parts += ["--duration", str(record.duration)]
    if record.language:
        parts += ["--language", _shell_quote(record.language)]
    if record.ref_audio:
        parts += ["--ref_audio", _shell_quote(record.ref_audio)]
    if record.ref_text:
        parts += ["--ref_text", _shell_quote(record.ref_text)]
    if record.instruct:
        parts += ["--instruct", _shell_quote(record.instruct)]
    if not record.denoise:
        parts += ["--denoise", "false"]
    if not record.postprocess_output:
        parts += ["--postprocess_output", "false"]
    if record.normalize_text:
        parts += ["--normalize_text", "true"]
    if record.batch_count > 1:
        parts += ["--batch_count", str(record.batch_count)]

    return " ".join(parts)


def _shell_quote(s: str) -> str:
    """对字符串进行 shell 转义（跨平台兼容）。"""
    if not s:
        return "''"
    # 简单情况：不含特殊字符，直接返回
    if all(c.isalnum() or c in "._-/" for c in s):
        return s
    # 含特殊字符：用双引号包裹并转义内部双引号
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
