"""URScript 兼容性静态检查（不需要机器人）。

这些约束来自 2026-11-08 在 UR5 / CB3 / PolyScope 3.15.8 真机上的踩坑：
在脚本端口下发时，控制器会把「整个顶层 `def` 的函数体」当程序运行，以下写法会编译失败：

1. 只允许**一个顶层函数定义**（helper 必须嵌套），否则控制器接受启动但立即终止；
2. `else` 必须写成 `else:`（冒号不能省）；
3. CB3 3.15 的 URScript 不支持 `for` 循环 / `abs()` / `max()`；
4. 不要出现顶层可执行语句（如顶层常量赋值、末尾调用），这些会改变程序结构语义。

真机之外无法编译 URScript，所以用静态检查把已知坑钉住。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "urscript" / "ur5_teleop.script"


def code_lines(text: str) -> list[str]:
    """去掉整行注释与空行（行内注释保留，但检查前会剥掉）。"""
    result = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append(raw.split("#")[0].rstrip())
    return result


class UR5ScriptCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")
        cls.lines = code_lines(cls.text)

    def test_single_top_level_def(self) -> None:
        top_level_defs = [line for line in self.lines if re.match(r"^def\s+\w+\s*\(", line)]
        self.assertEqual(len(top_level_defs), 1,
                         f"顶层 def 必须只有一个，实际：{top_level_defs}")

    def test_no_bare_else(self) -> None:
        bad = [(i, line) for i, line in enumerate(self.lines)
               if line.strip() == "else"]
        self.assertEqual(bad, [], f"else 必须写成 else:（带冒号），问题行：{bad}")

    def test_no_for_loop(self) -> None:
        bad = [(i, line) for i, line in enumerate(self.lines)
               if re.match(r"^\s*for\b", line)]
        self.assertEqual(bad, [], f"CB3 3.15 不支持 for 循环：{bad}")

    def test_no_unsupported_builtins(self) -> None:
        for name in ("abs", "max", "min"):
            pattern = re.compile(rf"(?<![\w.]){name}\s*\(")
            # 允许出现在注释里；code_lines 已去掉整行注释，行内注释也可能含函数名，先剥掉
            hits = [(i, line) for i, line in enumerate(self.lines)
                    if pattern.search(line.split("#")[0])]
            self.assertEqual(hits, [], f"CB3 3.15 不支持 {name}()：{hits}")

    def test_no_top_level_statements(self) -> None:
        """顶层除了注释/空行/唯一 def 之外不应有可执行语句。"""
        allowed_prefixes = ("def ",)
        for line in self.lines:
            stripped = line.strip()
            if stripped.startswith(allowed_prefixes) or stripped == "end":
                continue
            # 顶层行必须缩进（在 def 内部）；顶层出现非 def 语句即为问题
            self.assertTrue(line.startswith((" ", "\t")),
                            f"顶层不应有可执行语句：{line!r}")

    def test_register_contract_present(self) -> None:
        # 契约：double 0..5 目标位姿、int 1 命令字、int 0 心跳（通过 rtde_set_watchdog 字符串）
        for token in ("read_input_float_register(0)", "read_input_float_register(5)",
                      "read_input_integer_register(1)",
                      'rtde_set_watchdog("input_int_register_0"'):
            self.assertIn(token, self.text)

    def test_servoj_and_ik_present(self) -> None:
        self.assertIn("servoj(", self.text)
        self.assertIn("get_inverse_kin(", self.text)


if __name__ == "__main__":
    unittest.main()


TRAJ_SCRIPT = Path(__file__).resolve().parents[1] / "urscript" / "ur5_traj.script"


class UR5TrajScriptCompatTest(unittest.TestCase):
    """ur5_traj.script（离线关节轨迹播放）的结构约束，同 ur5_teleop.script 的 CB3 踩坑。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = TRAJ_SCRIPT.read_text(encoding="utf-8")
        cls.lines = code_lines(cls.text)

    def test_single_top_level_def(self) -> None:
        top_level_defs = [line for line in self.lines if re.match(r"^def\s+\w+\s*\(", line)]
        self.assertEqual(len(top_level_defs), 1,
                         f"顶层 def 必须只有一个，实际：{top_level_defs}")

    def test_no_bare_else(self) -> None:
        bad = [(i, line) for i, line in enumerate(self.lines) if line.strip() == "else"]
        self.assertEqual(bad, [], f"else 必须写成 else:（带冒号），问题行：{bad}")

    def test_no_for_loop(self) -> None:
        bad = [(i, line) for i, line in enumerate(self.lines)
               if re.match(r"^\s*for\b", line)]
        self.assertEqual(bad, [], f"CB3 3.15 不支持 for 循环：{bad}")

    def test_no_unsupported_builtins(self) -> None:
        for name in ("abs", "max", "min"):
            pattern = re.compile(rf"(?<![\w.]){name}\s*\(")
            hits = [line for line in self.lines if pattern.search(line.split("#")[0])]
            self.assertEqual(hits, [], f"CB3 3.15 不支持 {name}()：{hits}")

    def test_no_top_level_statements(self) -> None:
        for line in self.lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped == "end":
                continue
            self.assertTrue(line.startswith((" ", "\t")),
                            f"顶层不应有可执行语句：{line!r}")

    def test_no_rtde_watchdog(self) -> None:
        # CB3 3.15.8 实测：RTDE 看门狗触发会变成 C207A0 保护性停止；改用程序内心跳自检 + halt
        # 只查代码行：文件头注释里会提到为什么不用它
        hits = [line for line in self.lines if "rtde_set_watchdog" in line]
        self.assertEqual(hits, [], f"不应调用 RTDE 看门狗：{hits}")

    def test_traj_contract_present(self) -> None:
        for token in ("def ur5_traj(",
                      "read_input_float_register(0)", "read_input_float_register(5)",
                      "read_input_integer_register(0)", "read_input_integer_register(1)",
                      "movej(", "servoj(", "halt", "sync()"):
            self.assertIn(token, self.text)
