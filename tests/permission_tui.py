"""真实 DeepSeek + PTY 权限测试，会产生 API 用量。需 requirements-terminal.txt。"""
import re
from terminal_editing import Terminal


def command(terminal, text, expected):
    """发送控制命令，等待确认后再继续，避免把权限设置排到任务后面。"""
    start = len(terminal.raw)
    terminal.send(text + "\r")
    terminal.wait(lambda: expected in terminal.raw[start:] and terminal.line() == "你:")


def ask(terminal, seconds):
    """让真实模型申请等待工具，返回这次实际生成的审批编号。"""
    start = len(terminal.raw)
    terminal.send(f"请务必调用一次wait_for工具等待{seconds}秒；如果权限拒绝就停止，不要重试。简短回答工具结果。\r")
    terminal.wait(lambda: re.search(r"\[待批准 #(\d+)\]", terminal.raw[start:]), timeout=150)
    approval_id = re.search(r"\[待批准 #(\d+)\]", terminal.raw[start:]).group(1)
    return start, approval_id


def context(terminal):
    """读取真实 ContextMemory 中的工具结果，而不只依赖模型声称执行成功。"""
    start = len(terminal.raw)
    command(terminal, "/context", "[ContextMemory]")
    return terminal.raw[start:]


def run():
    """依次验证批准、单次拒绝、取消后旧编号失效和 deny 规则。"""
    terminal = Terminal()
    try:
        command(terminal, "/permission wait_for ask", "wait_for = ask")
        start, approval_id = ask(terminal, 1)
        command(terminal, f"/approve {approval_id}", "已批准")
        terminal.wait(lambda: "state=Completed" in terminal.raw[start:], timeout=150)
        snapshot = context(terminal)
        assert 'content: "已等待 1.0 秒", is_error: false' in snapshot
        print(f"PASS: ask → /approve {approval_id} → 真实 Tool 结果 已等待1秒", flush=True)

        start, rejected_id = ask(terminal, 2)
        assert rejected_id != approval_id
        command(terminal, f"/deny {rejected_id}", "已拒绝")
        terminal.wait(lambda: "state=Completed" in terminal.raw[start:], timeout=150)
        snapshot = context(terminal)
        assert "权限拒绝或未获批准: wait_for" in snapshot
        assert "已等待 2.0 秒" not in snapshot
        print(f"PASS: 下一轮仍需批准；/deny {rejected_id} → Tool 权限错误", flush=True)

        before = snapshot
        start, stale_id = ask(terminal, 3)
        command(terminal, "/cancel", "state=Cancelled")
        command(terminal, f"/approve {stale_id}", "没有该编号的有效审批")
        after = context(terminal)
        before_count = re.search(r"messages=(\d+)", before).group(1)
        assert re.search(r"messages=(\d+)", after).group(1) == before_count
        assert "已等待 3.0 秒" not in after
        print(f"PASS: /cancel 回滚；过期 /approve {stale_id} 不执行", flush=True)

        command(terminal, "/permission wait_for deny", "wait_for = deny")
        start = len(terminal.raw)
        terminal.send("请务必调用一次wait_for工具等待4秒。如果拒绝则停止。\r")
        terminal.wait(lambda: "state=Completed" in terminal.raw[start:], timeout=150)
        assert "[待批准 #" not in terminal.raw[start:]
        snapshot = context(terminal)
        assert "已等待 4.0 秒" not in snapshot
        assert "权限拒绝或未获批准: wait_for" in snapshot
        command(terminal, "/reset", "记忆已清空")
        command(terminal, "/permissions", "wait_for: deny")
        print("PASS: deny 不弹审批、不执行；/reset 保留权限", flush=True)
        terminal.close()
    finally:
        terminal.cleanup()


if __name__ == "__main__":
    run()
