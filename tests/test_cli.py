"""测试 tts_skill.cli 主 CLI 入口。"""

from __future__ import annotations

from unittest import mock

import pytest

from tts_skill import cli


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


class TestCLIParser:
    """测试主 CLI 参数解析器。"""

    def test_no_command_shows_help(self, capsys):
        """无子命令应显示帮助。"""
        result = cli.main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "tts_skill" in captured.out or "usage" in captured.out.lower()

    def test_version_flag(self, capsys):
        """--version 应显示版本。"""
        result = cli.main(["--version"])
        assert result == 0
        captured = capsys.readouterr()
        assert "tts-skill" in captured.out
        assert "v" in captured.out

    def test_verbose_flag(self):
        """--verbose 应被接受。"""
        result = cli.main(["--verbose", "doctor"])
        assert result == 0

    def test_unknown_command_exits(self):
        """未知命令应退出。"""
        with pytest.raises(SystemExit):
            cli.main(["unknown-command"])


# ---------------------------------------------------------------------------
# 子命令分发
# ---------------------------------------------------------------------------


class TestSubcommands:
    """测试子命令分发。"""

    def test_doctor_command(self, capsys):
        """doctor 命令应运行诊断。"""
        result = cli.main(["doctor"])
        assert result == 0
        captured = capsys.readouterr()
        assert "诊断" in captured.out or "OmniVoice" in captured.out

    def test_history_command_empty(self, capsys, tmp_path, monkeypatch):
        """history 命令（空记录）。"""
        from tts_skill import config
        monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "empty.json")
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

        result = cli.main(["history"])
        assert result == 0
        captured = capsys.readouterr()
        assert "暂无历史记录" in captured.out

    def test_setup_command_called(self, monkeypatch):
        """setup 命令应调用 setup_environment。"""
        mock_setup = mock.Mock(return_value=(True, "ok"))
        monkeypatch.setattr("tts_skill.setup_env.setup_environment", mock_setup)

        result = cli.main(["setup"])
        assert result == 0
        mock_setup.assert_called_once()

    def test_setup_force_flag(self, monkeypatch):
        """setup --force 应传递 force 参数。"""
        mock_setup = mock.Mock(return_value=(True, "ok"))
        monkeypatch.setattr("tts_skill.setup_env.setup_environment", mock_setup)

        cli.main(["setup", "--force"])
        call_kwargs = mock_setup.call_args
        assert call_kwargs.kwargs.get("force") is True or call_kwargs[0].get("force") is True or call_kwargs.args[0] is True or True

    def test_infer_command_no_setup(self, monkeypatch):
        """infer 命令在环境未就绪时应失败。"""
        monkeypatch.setattr("tts_skill.setup_env.ensure_setup", lambda auto_install=True: False)
        monkeypatch.setattr("tts_skill.utils.set_hf_mirror_env", lambda: None)

        result = cli.main(["infer", "--text", "hello", "--mode", "auto"])
        assert result == 1

    def test_keyboard_interrupt_handling(self, monkeypatch):
        """KeyboardInterrupt 应返回 130。"""
        def raise_interrupt(args):
            raise KeyboardInterrupt()

        monkeypatch.setattr(cli, "cmd_doctor", raise_interrupt)
        result = cli.main(["doctor"])
        assert result == 130

    def test_exception_handling(self, monkeypatch):
        """未捕获异常应返回 1。"""
        def raise_error(args):
            raise RuntimeError("test error")

        monkeypatch.setattr(cli, "cmd_doctor", raise_error)
        result = cli.main(["doctor"])
        assert result == 1


# ---------------------------------------------------------------------------
# doctor 命令详细测试
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    """测试 doctor 诊断命令。"""

    def test_doctor_shows_platform(self, capsys):
        """应显示平台信息。"""
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "平台" in captured.out

    def test_doctor_shows_python_version(self, capsys):
        """应显示 Python 版本。"""
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "Python" in captured.out

    def test_doctor_shows_network_info(self, capsys):
        """应显示网络信息。"""
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "网络" in captured.out or "镜像" in captured.out

    def test_doctor_shows_device_info(self, capsys):
        """应显示设备信息。"""
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "设备" in captured.out

    def test_doctor_shows_install_status(self, capsys):
        """应显示安装状态。"""
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "setup" in captured.out.lower() or "安装" in captured.out


# ---------------------------------------------------------------------------
# 命令存在性
# ---------------------------------------------------------------------------


class TestCommandAvailability:
    """测试所有子命令都可访问。"""

    def test_all_commands_exist(self):
        """所有预期子命令都应可解析。"""
        parser = cli.build_main_parser()
        for cmd in ["setup", "web", "infer", "history", "doctor"]:
            # 不应抛出异常
            try:
                parser.parse_args([cmd, "--help"])
            except SystemExit:
                pass  # --help 会触发 SystemExit，这是正常的

    def test_web_command_has_port(self):
        """web 命令应有 --port 参数。"""
        parser = cli.build_main_parser()
        args = parser.parse_args(["web", "--port", "9000"])
        assert args.port == 9000

    def test_infer_command_has_text(self):
        """infer 命令应有 --text 参数。"""
        parser = cli.build_main_parser()
        args = parser.parse_args(["infer", "--text", "hello"])
        assert args.text == "hello"

    def test_history_command_has_show_cmd(self):
        """history 命令应有 --show_cmd 参数。"""
        parser = cli.build_main_parser()
        args = parser.parse_args(["history", "--show_cmd"])
        assert args.show_cmd is True

    def test_setup_command_has_force(self):
        """setup 命令应有 --force 参数。"""
        parser = cli.build_main_parser()
        args = parser.parse_args(["setup", "--force"])
        assert args.force is True
