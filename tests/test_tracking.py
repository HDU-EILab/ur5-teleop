"""``ur5_teleop.tracking`` 的帧校验与帧来源单测（stdlib unittest）。"""

import json
import pathlib
import socket
import tempfile
import time
import unittest

from ur5_teleop import tracking

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "tests" / "data" / "sample_frames.jsonl"


def make_frame_dict(**overrides):
    """构造一帧合法的 probe.py 格式数据。"""
    frame = {
        "pico_time_ns": 1_000_000_000,
        "bridge_time_ns": 1_760_000_000_000_000_000,
        "left": [-0.2, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0],
        "right": [0.2, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0],
        "head": [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0],
        "left_grip": 0.1,
        "right_grip": 0.9,
        "left_trigger": 0.0,
        "right_trigger": 0.0,
    }
    frame.update(overrides)
    return frame


class ParseFrameTest(unittest.TestCase):
    def test_parse_dict_bytes_and_str(self):
        frame_dict = make_frame_dict()
        for packet in (frame_dict,
                       json.dumps(frame_dict).encode("utf-8"),
                       json.dumps(frame_dict)):
            with self.subTest(packet_type=type(packet).__name__):
                frame = tracking.parse_frame(packet)
                self.assertEqual(frame.pico_time_ns, 1_000_000_000)
                self.assertEqual(frame.pose("right"), (0.2, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0))
                self.assertAlmostEqual(frame.grip("right"), 0.9)
                self.assertAlmostEqual(frame.grip("left"), 0.1)
                self.assertEqual(frame.pose("head")[:2], (0.0, 1.6))

    def test_unknown_side_raises(self):
        frame = tracking.parse_frame(make_frame_dict())
        with self.assertRaises(ValueError):
            frame.pose("controller")
        with self.assertRaises(ValueError):
            frame.grip("head")

    def assert_invalid(self, packet):
        with self.assertRaises(tracking.InvalidFrame):
            tracking.parse_frame(packet)

    def test_invalid_json(self):
        self.assert_invalid(b"{not json")
        self.assert_invalid(b"\xff\xfe")

    def test_frame_must_be_object(self):
        self.assert_invalid(json.dumps([1, 2, 3]))
        self.assert_invalid(123)

    def test_invalid_timestamp(self):
        self.assert_invalid(make_frame_dict(pico_time_ns=0))
        self.assert_invalid(make_frame_dict(pico_time_ns=-5))
        self.assert_invalid(make_frame_dict(pico_time_ns="1000"))
        self.assert_invalid(make_frame_dict(pico_time_ns=1.5))
        self.assert_invalid(make_frame_dict(pico_time_ns=True))
        frame = make_frame_dict()
        del frame["pico_time_ns"]
        self.assert_invalid(frame)

    def test_invalid_pose(self):
        frame = make_frame_dict()
        del frame["left"]
        self.assert_invalid(frame)
        self.assert_invalid(make_frame_dict(right=[0.2, 1.2, -0.3, 0.0, 0.0, 1.0]))
        self.assert_invalid(make_frame_dict(right=[0.2, 1.2, -0.3, 0.0, 0.0, 0.0, "x"]))
        self.assert_invalid(make_frame_dict(right=[0.2, float("nan"), -0.3, 0.0, 0.0, 0.0, 1.0]))
        self.assert_invalid(make_frame_dict(right=[0.2, 1.2, -0.3, 0.0, 0.0, 0.0, float("inf")]))

    def test_invalid_quaternion_norm(self):
        self.assert_invalid(make_frame_dict(
            head=[0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 2.0]))
        self.assert_invalid(make_frame_dict(
            head=[0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 0.1]))

    def test_untracked_pose_rejected(self):
        # 位置全 0 + 单位四元数 = PICO 还没拿到该设备的追踪数据
        self.assert_invalid(make_frame_dict(
            left=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    def test_invalid_grip(self):
        frame = make_frame_dict()
        del frame["left_grip"]
        self.assert_invalid(frame)
        self.assert_invalid(make_frame_dict(left_grip=1.5))
        self.assert_invalid(make_frame_dict(right_grip=-0.1))
        self.assert_invalid(make_frame_dict(right_grip=float("nan")))
        self.assert_invalid(make_frame_dict(right_grip="high"))


class UdpFrameSourceTest(unittest.TestCase):
    def setUp(self):
        self.source = tracking.UdpFrameSource(0, host="127.0.0.1")
        self.addCleanup(self.source.close)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(self.sender.close)
        self.address = ("127.0.0.1", self.source.port)

    def send(self, packet):
        if isinstance(packet, dict):
            packet = json.dumps(packet).encode("utf-8")
        self.sender.sendto(packet, self.address)

    def poll_until_frame(self, timeout_s=0.5):
        """UDP 回环投递在 macOS 上是异步的，轮询等待一小段时间。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.source.poll()
            if frame is not None:
                return frame
            time.sleep(0.001)
        return None

    def poll_expect_none(self, wait_s=0.02):
        """等一小段时间后断言没有可用新帧（丢弃 invalid/重复帧后返回 None）。"""
        time.sleep(wait_s)
        return self.source.poll()

    def test_poll_receives_valid_frame(self):
        self.assertIsNone(self.source.last_receive)
        self.send(make_frame_dict())
        frame = self.poll_until_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.pico_time_ns, 1_000_000_000)
        self.assertIsNotNone(self.source.last_receive)
        # 没有新包时返回 None
        self.assertIsNone(self.poll_expect_none())

    def test_invalid_and_stale_packets_are_skipped(self):
        self.send(make_frame_dict(pico_time_ns=2_000))
        self.assertIsNotNone(self.poll_until_frame())
        self.send(b"not json")
        self.assertIsNone(self.poll_expect_none())
        self.assertEqual(self.source.invalid_packets, 1)
        # 时间戳不单调（更旧）的帧应丢弃
        self.send(make_frame_dict(pico_time_ns=1_500))
        self.assertIsNone(self.poll_expect_none())
        self.assertEqual(self.source.last_pico_time_ns, 2_000)
        # 更新的帧恢复接收
        self.send(make_frame_dict(pico_time_ns=3_000))
        frame = self.poll_until_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.pico_time_ns, 3_000)

    def test_is_stale(self):
        self.assertTrue(self.source.is_stale(0.2))  # 从未收到
        self.send(make_frame_dict())
        self.assertIsNotNone(self.poll_until_frame())
        self.assertFalse(self.source.is_stale(0.2))
        self.assertFalse(self.source.is_stale(1.0))
        self.assertTrue(self.source.is_stale(0.2, now=self.source.last_receive + 0.3))

    def test_close_is_idempotent(self):
        self.source.close()
        self.source.close()
        self.assertFalse(self.source.exhausted)

    def test_satisfies_frame_source_protocol(self):
        self.assertIsInstance(self.source, tracking.FrameSource)


class ReplayFrameSourceTest(unittest.TestCase):
    def test_from_jsonl_reads_sample_recording(self):
        source = tracking.ReplayFrameSource.from_jsonl(SAMPLE_PATH)
        self.assertIsInstance(source, tracking.FrameSource)
        stamps = []
        while not source.exhausted:
            frame = source.poll()
            self.assertIsNotNone(frame)
            stamps.append(frame.pico_time_ns)
        self.assertEqual(len(stamps), 20)
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(set(stamps)), 20)
        self.assertIsNone(source.poll())  # 用完后一直返回 None
        self.assertIsNotNone(source.last_receive)

    def test_exhausted_only_after_last_frame(self):
        source = tracking.ReplayFrameSource([tracking.parse_frame(make_frame_dict())])
        self.assertFalse(source.exhausted)
        self.assertIsNotNone(source.poll())
        self.assertTrue(source.exhausted)
        self.assertIsNone(source.poll())

    def test_bad_line_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.jsonl"
            good = json.dumps(make_frame_dict())
            path.write_text(good + "\n{bad json\n", encoding="utf-8")
            with self.assertRaises(tracking.InvalidFrame) as context:
                tracking.ReplayFrameSource.from_jsonl(path)
            self.assertIn(":2:", str(context.exception))


if __name__ == "__main__":
    unittest.main()

    def test_button_and_axis_parsing(self):
        raw = {
            "pico_time_ns": 123,
            "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "right": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            "head": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
            "left_grip": 0.1, "right_grip": 0.9,
            "left_trigger": 0.2, "right_trigger": 0.7,
            "buttons": {"a": False, "b": True, "axis_right_click": False},
            "axis": {"left": [0.1, -0.2], "right": [0.0, 0.5]},
        }
        frame = parse_frame(raw)
        self.assertTrue(frame.button("b"))
        self.assertFalse(frame.button("a"))
        self.assertFalse(frame.button("unknown"))
        self.assertAlmostEqual(frame.axis["left"][0], 0.1)
        self.assertAlmostEqual(frame.axis["right"][1], 0.5)

    def test_missing_buttons_default_to_empty(self):
        raw = {
            "pico_time_ns": 123,
            "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "right": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            "head": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
            "left_grip": 0.1, "right_grip": 0.9,
            "left_trigger": 0.0, "right_trigger": 0.0,
        }
        frame = parse_frame(raw)
        self.assertFalse(frame.button("b"))
        self.assertEqual(frame.axis, {})
