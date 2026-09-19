"""真实 PTY 行编辑回归：需 pyte；--live 额外调用 DeepSeek，产生 API 用量。

运行：cargo build && python3 tests/terminal_editing.py [--live]
"""
import argparse
import fcntl
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

import pyte

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
BINARY = TARGET_DIR / "debug/asteria-agent"


class Terminal:
    """真实运行 Asteria，将 ANSI 输出解析为屏幕，检查显示而不只检查原始字节。"""

    def __init__(self, columns=80, env_overrides=None):
        self.master, slave = pty.openpty()
        self.original_mode = termios.tcgetattr(self.master)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, columns, 0, 0))
        environment = dict(os.environ, TERM="xterm-256color")
        self.session_dir = Path(tempfile.mkdtemp(prefix="asteria-tui-test-"))
        environment["ASTERIA_SESSION_PATH"] = str(self.session_dir)
        environment["ASTERIA_NO_PROJECT_MCP"] = "1"
        environment["ASTERIA_NO_RAG"] = "1"
        environment.update(env_overrides or {})

        def controlling_terminal():
            """为子进程建立独立会话和控制终端，避免向其他进程发送信号。"""
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.process = subprocess.Popen(
            [str(BINARY)], cwd=ROOT, env=environment,
            stdin=slave, stdout=slave, stderr=slave, preexec_fn=controlling_terminal,
        )
        os.close(slave)
        self.screen = pyte.Screen(columns, 40)
        # 模拟真终端对光标位置查询的回复；否则编辑器无法判断重绘位置。
        self.screen.write_process_input = lambda data: os.write(self.master, data.encode())
        self.stream = pyte.ByteStream(self.screen)
        self.raw = ""
        self.wait(self.at_prompt)

    def line(self):
        """读取光标所在行的最终显示内容。"""
        return self.screen.display[self.screen.cursor.y].rstrip()

    def at_prompt(self, text=""):
        """空闲时单行 User 提示，输入紧跟在标签后面。"""
        return self.line() == f"User：{text}".rstrip()

    def send(self, text):
        """一次性写入整段 UTF-8 和控制键，覆盖中文连输/粘贴场景。"""
        os.write(self.master, text.encode())

    def wait(self, predicate, timeout=8):
        """驱动终端模拟器直到断言可见，失败时打印当前屏幕供诊断。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            ready, _, _ = select.select([self.master], [], [], 0.02)
            if ready:
                try:
                    data = os.read(self.master, 65536)
                except OSError:
                    break
                self.raw += data.decode("utf-8", errors="replace")
                self.stream.feed(data)
        raise AssertionError("等待超时，屏幕：\n" + "\n".join(self.screen.display))

    def edited_command(self, typed, editing, command, output):
        """检查退格后的可见输入和实际执行的命令一致。"""
        self.send(typed + editing + command)
        self.wait(lambda: self.at_prompt(command))
        start = len(self.raw)
        self.send("\r")
        self.wait(lambda: output in self.raw[start:] and self.at_prompt())
        assert f"› {command}" in self.raw[start:], self.raw[start:]
        assert any(command in line for line in self.screen.display), (
            "提交后的命令回显被后续输出擦掉了：\n" + "\n".join(self.screen.display)
        )
        assert "[处理中]" not in self.raw[start:], "命令残留前缀，误发给模型"

    def close(self):
        """验证 /exit 正常退出，且终端的回显与规范输入模式已恢复。"""
        try:
            self.send("\x15/exit\r")
            self.wait(lambda: self.process.poll() is not None)
        except AssertionError:
            if self.process.poll() is None:
                raise
        assert self.process.wait(timeout=2) == 0
        mask = termios.ECHO | termios.ICANON | termios.ISIG
        assert termios.tcgetattr(self.master)[3] & mask == self.original_mode[3] & mask
        os.close(self.master)
        shutil.rmtree(self.session_dir, ignore_errors=True)

    def cleanup(self):
        """只在失败时回收本测试启动的进程。"""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        shutil.rmtree(self.session_dir, ignore_errors=True)


def check_editing():
    """不调用模型，复现中文标点、中文词语、组合字符与光标移动后的命令输入。"""
    for width in (80, 24):
        terminal = Terminal(width)
        try:
            terminal.edited_command("、、", "\x7f" * 2, "/usage", "尚未执行 Turn")
            terminal.edited_command("开始处理", "\x7f" * 4, "/context", "messages=0")
            terminal.edited_command("中文e\u0301", "\x7f" * 3, "/usage", "尚未执行 Turn")
            terminal.edited_command("中文", "\x1b[D\x7f\x1b[C\x7f", "/usage", "尚未执行 Turn")
            terminal.edited_command("中文标点" * 8, "\x7f" * 32, "/context", "messages=0")
            terminal.edited_command("\x1b[200~、、\x1b[201~", "\x7f" * 2, "/usage", "尚未执行 Turn")
            terminal.send("残留中文\x03")
            terminal.wait(lambda: "当前没有运行中的 Turn" in terminal.raw and terminal.at_prompt())
            terminal.close()
            print(f"PASS: {width}列，中文/标点/组合字符/光标/跨行/粘贴/Ctrl+C/退出恢复")
        finally:
            terminal.cleanup()


def check_live():
    """真实 DeepSeek 回答期间保留输入草稿，并验证 /cancel 与 SIGINT。"""
    terminal = Terminal()
    try:
        start = len(terminal.raw)
        terminal.send("请用300字介绍Rust所有权。\r")
        terminal.wait(lambda: "正在处理" in terminal.raw[start:])
        terminal.send("中文草稿、、")
        terminal.wait(lambda: "tokens" in terminal.raw[start:], timeout=150)
        terminal.wait(lambda: terminal.at_prompt("中文草稿、、"))
        start = len(terminal.raw)
        terminal.edited_command("", "\x7f" * 6, "/usage", "上下文消息：2")
        assert "=== 执行失败 ===" not in terminal.raw[start:]
        print("PASS: 真实回答与用量输出后，中文草稿保持完整；删除草稿后 /usage 正确执行")
        for cancel in ("/cancel\r", "\x03", "SIGINT"):
            start = len(terminal.raw)
            terminal.send("请写3000字的Rust教程。\r")
            terminal.wait(lambda: "正在处理" in terminal.raw[start:])
            if cancel == "SIGINT":
                os.kill(terminal.process.pid, signal.SIGINT)
            else:
                terminal.send(cancel)
            terminal.wait(lambda: "Turn 已取消" in terminal.raw[start:] and terminal.at_prompt())
            context_start = len(terminal.raw)
            terminal.send("/context\r")
            terminal.wait(lambda: "messages=2" in terminal.raw[context_start:] and terminal.at_prompt())
            print(f"PASS: 真实请求 {cancel!r} 取消，原始历史仍为2条")
        terminal.close()
    finally:
        terminal.cleanup()


def check_retry_output():
    """实际连接不可用代理，检查重试与失败信息不会破坏输入草稿。"""
    terminal = Terminal(env_overrides={
        "HTTPS_PROXY": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
        "ALL_PROXY": "http://127.0.0.1:1", "all_proxy": "http://127.0.0.1:1",
        "NO_PROXY": "", "no_proxy": "",
    })
    try:
        terminal.send("连接故障测试\r")
        terminal.wait(lambda: "↻ Step" in terminal.raw)
        terminal.send("中文草稿、、")
        terminal.wait(lambda: "=== 执行失败 ===" in terminal.raw)
        terminal.wait(lambda: terminal.at_prompt("中文草稿、、"))
        terminal.edited_command("", "\x7f" * 6, "/usage", "上下文消息：0")
        terminal.close()
        print("PASS: 真实连接失败、重试及失败报告输出后，中文输入行保持完整")
    finally:
        terminal.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="额外调用真实 DeepSeek")
    args = parser.parse_args()
    check_editing()
    check_retry_output()
    if args.live:
        check_live()
