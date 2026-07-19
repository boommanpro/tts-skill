"""测试 tts_skill.config 配置与历史记录管理。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from tts_skill import config
from tts_skill.config import (
    GenerationRecord,
    _shell_quote,
    add_record,
    build_reproducible_command,
    create_record,
    delete_record,
    get_record,
    load_history,
    save_history,
    update_record_name,
)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    """每个测试使用独立的历史记录文件。"""
    history_file = tmp_path / "history.json"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(config, "HISTORY_FILE", history_file)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    yield history_file


@pytest.fixture
def sample_record():
    """创建一个示例记录数据。"""
    return {
        "id": "abc12345",
        "name": "测试记录",
        "timestamp": "2026-07-19 10:00:00",
        "mode": "auto",
        "text": "你好世界",
        "seed": 42,
        "num_step": 32,
        "speed": 1.0,
        "batch_count": 1,
        "batch_seeds": [42],
        "output_files": ["/tmp/test.wav"],
    }


# ---------------------------------------------------------------------------
# GenerationRecord 数据模型
# ---------------------------------------------------------------------------


class TestGenerationRecord:
    """测试 GenerationRecord 数据类。"""

    def test_create_record_minimal(self):
        """最小字段创建记录。"""
        record = GenerationRecord(
            id="test1",
            name="test",
            timestamp="2026-07-19 10:00:00",
            mode="auto",
            text="hello",
        )
        assert record.id == "test1"
        assert record.mode == "auto"
        assert record.seed == 0  # 默认值
        assert record.batch_count == 1
        assert record.output_files == []

    def test_to_dict(self, sample_record):
        """to_dict 应包含所有字段。"""
        record = GenerationRecord.from_dict(sample_record)
        d = record.to_dict()
        assert d["id"] == "abc12345"
        assert d["text"] == "你好世界"
        assert d["seed"] == 42

    def test_from_dict_with_extra_fields(self, sample_record):
        """from_dict 应忽略多余字段。"""
        sample_record["extra_field"] = "should be ignored"
        record = GenerationRecord.from_dict(sample_record)
        assert record.id == "abc12345"
        assert not hasattr(record, "extra_field")

    def test_from_dict_with_missing_fields(self):
        """from_dict 应填充缺失字段为默认值。"""
        record = GenerationRecord.from_dict({
            "id": "test",
            "name": "test",
            "timestamp": "2026-07-19",
            "mode": "auto",
            "text": "hello",
        })
        assert record.seed == 0
        assert record.batch_count == 1
        assert record.batch_seeds == []


# ---------------------------------------------------------------------------
# 历史记录 CRUD
# ---------------------------------------------------------------------------


class TestHistoryCRUD:
    """测试历史记录增删改查。"""

    def test_load_empty_history(self, isolated_history):
        """空历史应返回空列表。"""
        records = load_history()
        assert records == []

    def test_add_and_load_record(self, isolated_history):
        """添加记录后应能加载。"""
        record = GenerationRecord(
            id="r1",
            name="record1",
            timestamp="2026-07-19 10:00:00",
            mode="auto",
            text="hello",
        )
        add_record(record)

        records = load_history()
        assert len(records) == 1
        assert records[0].id == "r1"

    def test_load_history_sorted_by_time_desc(self, isolated_history):
        """历史记录应按时间倒序。"""
        r1 = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 09:00:00",
            mode="auto", text="1",
        )
        r2 = GenerationRecord(
            id="r2", name="r2", timestamp="2026-07-19 10:00:00",
            mode="auto", text="2",
        )
        save_history([r1, r2])

        records = load_history()
        assert records[0].id == "r2"  # 更晚的在前
        assert records[1].id == "r1"

    def test_get_record_by_id(self, isolated_history):
        """按 ID 获取记录。"""
        record = GenerationRecord(
            id="target", name="target", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello",
        )
        add_record(record)

        found = get_record("target")
        assert found is not None
        assert found.id == "target"

    def test_get_record_not_found(self, isolated_history):
        """获取不存在的记录应返回 None。"""
        assert get_record("nonexistent") is None

    def test_update_record_name(self, isolated_history):
        """更新记录名称。"""
        record = GenerationRecord(
            id="r1", name="old_name", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello",
        )
        add_record(record)

        assert update_record_name("r1", "new_name") is True
        found = get_record("r1")
        assert found.name == "new_name"

    def test_update_name_not_found(self, isolated_history):
        """更新不存在的记录应返回 False。"""
        assert update_record_name("nonexistent", "name") is False

    def test_delete_record(self, isolated_history, tmp_path):
        """删除记录。"""
        # 创建一个临时输出文件
        output_file = tmp_path / "test.wav"
        output_file.write_text("dummy")

        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello",
            output_files=[str(output_file)],
        )
        add_record(record)

        assert delete_record("r1") is True
        assert get_record("r1") is None
        assert not output_file.exists()  # 输出文件也应被删除

    def test_delete_record_not_found(self, isolated_history):
        """删除不存在的记录应返回 False。"""
        assert delete_record("nonexistent") is False


# ---------------------------------------------------------------------------
# create_record 工厂函数
# ---------------------------------------------------------------------------


class TestCreateRecord:
    """测试 create_record 工厂函数。"""

    def test_create_record_auto_mode(self, isolated_history):
        """创建 auto 模式记录。"""
        record = create_record(
            mode="auto",
            text="你好",
            seed=42,
            batch_count=1,
            output_files=["/tmp/out.wav"],
        )
        assert record.id  # 自动生成
        assert record.mode == "auto"
        assert record.seed == 42
        assert record.output_files == ["/tmp/out.wav"]

        # 应已保存到历史
        loaded = load_history()
        assert len(loaded) == 1
        assert loaded[0].id == record.id

    def test_create_record_clone_mode(self, isolated_history):
        """创建 clone 模式记录。"""
        record = create_record(
            mode="clone",
            text="hello",
            ref_audio="/tmp/ref.wav",
            ref_text="reference",
            seed=100,
        )
        assert record.mode == "clone"
        assert record.ref_audio == "/tmp/ref.wav"
        assert record.ref_text == "reference"

    def test_create_record_design_mode(self, isolated_history):
        """创建 design 模式记录。"""
        record = create_record(
            mode="design",
            text="hello",
            instruct="male, british accent",
            seed=200,
        )
        assert record.mode == "design"
        assert record.instruct == "male, british accent"

    def test_create_record_batch(self, isolated_history):
        """创建批量生成记录。"""
        record = create_record(
            mode="auto",
            text="test",
            seed=42,
            batch_count=5,
            batch_seeds=[42, 43, 44, 45, 46],
        )
        assert record.batch_count == 5
        assert record.batch_seeds == [42, 43, 44, 45, 46]

    def test_create_record_with_custom_name(self, isolated_history):
        """使用自定义名称。"""
        record = create_record(
            mode="auto",
            text="test",
            name="我的自定义名称",
        )
        assert record.name == "我的自定义名称"

    def test_create_record_auto_name(self, isolated_history):
        """无名称时自动生成。"""
        record = create_record(mode="auto", text="test")
        assert "auto" in record.name


# ---------------------------------------------------------------------------
# 可复现命令生成
# ---------------------------------------------------------------------------


class TestReproducibleCommand:
    """测试可复现命令生成。"""

    def test_basic_command(self, isolated_history):
        """基本命令结构。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello", seed=42,
        )
        cmd = build_reproducible_command(record)
        assert "tts_skill" in cmd
        assert "infer" in cmd
        assert "--text" in cmd
        assert "hello" in cmd
        assert "--mode" in cmd
        assert "auto" in cmd
        assert "--seed" in cmd
        assert "42" in cmd

    def test_command_with_clone_params(self, isolated_history):
        """clone 模式命令。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="clone", text="hello",
            ref_audio="/tmp/ref.wav", ref_text="reference",
            seed=42,
        )
        cmd = build_reproducible_command(record)
        assert "--ref_audio" in cmd
        assert "/tmp/ref.wav" in cmd
        assert "--ref_text" in cmd
        assert "reference" in cmd

    def test_command_with_design_params(self, isolated_history):
        """design 模式命令。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="design", text="hello",
            instruct="male, british accent",
            seed=42,
        )
        cmd = build_reproducible_command(record)
        assert "--instruct" in cmd
        assert "male" in cmd

    def test_command_with_batch_count(self, isolated_history):
        """批量生成命令。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello", seed=42,
            batch_count=5,
        )
        cmd = build_reproducible_command(record)
        assert "--batch_count" in cmd
        assert "5" in cmd

    def test_command_with_special_chars(self, isolated_history):
        """包含特殊字符的文本。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="auto", text='hello "world" & friends',
            seed=42,
        )
        cmd = build_reproducible_command(record)
        # 应包含转义后的文本
        assert "hello" in cmd

    def test_command_includes_python_executable(self, isolated_history):
        """命令应包含 python 可执行文件。"""
        record = GenerationRecord(
            id="r1", name="r1", timestamp="2026-07-19 10:00:00",
            mode="auto", text="hello", seed=42,
        )
        cmd = build_reproducible_command(record)
        assert "-m" in cmd
        assert "tts_skill" in cmd


# ---------------------------------------------------------------------------
# shell 转义
# ---------------------------------------------------------------------------


class TestShellQuote:
    """测试 shell 转义函数。"""

    def test_simple_string(self):
        """简单字符串不需要转义。"""
        assert _shell_quote("hello") == "hello"

    def test_empty_string(self):
        """空字符串。"""
        assert _shell_quote("") == "''"

    def test_string_with_spaces(self):
        """含空格的字符串。"""
        result = _shell_quote("hello world")
        assert result.startswith('"')
        assert result.endswith('"')
        assert "hello world" in result

    def test_string_with_quotes(self):
        """含引号的字符串。"""
        result = _shell_quote('hello "world"')
        assert '\\"' in result

    def test_path_string(self):
        """路径字符串。"""
        assert _shell_quote("/tmp/file.wav") == "/tmp/file.wav"

    def test_chinese_string(self):
        """中文字符串。"""
        result = _shell_quote("你好世界")
        # 中文不是字母数字，应该被引号包裹
        assert result.startswith('"') or result == "你好世界"
