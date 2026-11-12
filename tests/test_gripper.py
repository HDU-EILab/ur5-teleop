"""大寰 PGC 夹爪协议与控制器单测（不连真机）。"""

import time
import unittest

from ur5_teleop.gripper import (
    Gripper,
    GripperError,
    POSITION_MAX,
    build_read,
    build_write,
    crc16,
)


class FakeTransport:
    """假的串口传输：记录写出的帧，按队列返回读响应。"""

    def __init__(self, responses=None):
        self.written: list[bytes] = []
        self.responses = list(responses or [])
        self.closed = False

    def reset_input_buffer(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))
        # 只有写寄存器（FC06/FC10）会回显；读请求（FC03）不回显
        if len(data) >= 2 and data[1] in (0x06, 0x10):
            self.responses.insert(0, bytes(data))

    def read(self, size: int) -> bytes:
        if not self.responses:
            return b""
        return self.responses.pop(0)[:size]

    def close(self) -> None:
        self.closed = True


def status_response(slave_id=1, state=1, position=250):
    data = bytes([slave_id, 0x03, 4]) + state.to_bytes(2, "big") + position.to_bytes(2, "big")
    return data + crc16(data).to_bytes(2, "little")


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class CrcTest(unittest.TestCase):
    def test_known_frame_crc(self):
        # 标准 Modbus RTU 例子：01 03 00 00 00 01 → CRC 0x84 0x0A
        self.assertEqual(build_read(1, 0x0000, 1).hex(), "010300000001840a")

    def test_crc16_matches_known_value(self):
        self.assertEqual(crc16(bytes.fromhex("010300000001")), 0x0A84)


class FrameTest(unittest.TestCase):
    def test_build_write_single_register(self):
        frame = build_write(1, 0x0103, [500])
        self.assertEqual(frame[0], 1)
        self.assertEqual(frame[1], 0x06)
        self.assertEqual(frame[2:4], bytes([0x01, 0x03]))
        self.assertEqual(frame[4:6], (500).to_bytes(2, "big"))
        self.assertEqual(crc16(frame[:-2]), int.from_bytes(frame[-2:], "little"))

    def test_build_write_multiple_registers(self):
        frame = build_write(2, 0x0200, [1, 2])
        self.assertEqual(frame[1], 0x10)
        self.assertEqual(frame[4:6], bytes([0x00, 0x02]))
        self.assertEqual(frame[6], 4)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            build_write(1, 0x0103, [])
        with self.assertRaises(ValueError):
            build_write(1, 0x0103, [0x10000])
        with self.assertRaises(ValueError):
            build_read(1, 0x0201, 0)


class GripperControllerTest(unittest.TestCase):
    def _make(self, responses):
        transport = FakeTransport(responses)
        gripper = Gripper(transport=transport, rate_hz=200.0, read_hz=50.0)
        return gripper, transport

    def test_connect_sends_enable_speed_force_and_reads_status(self):
        gripper, transport = self._make([status_response()] * 20)
        try:
            description = gripper.connect()
            self.assertIn("位置=250", description)
            written = [frame.hex() for frame in transport.written]
            self.assertTrue(any(f.startswith("01060106") for f in written), written)
            self.assertTrue(any(f.startswith("01060104") for f in written), written)
            self.assertTrue(any(f.startswith("01060101") for f in written), written)
            self.assertEqual(gripper.position_raw, 250)
            self.assertEqual(gripper.grip_state, 1)
            self.assertTrue(gripper.healthy)
        finally:
            gripper.close()
        self.assertTrue(transport.closed)

    def test_set_position_writes_target_register(self):
        gripper, transport = self._make([status_response()] * 50)
        try:
            gripper.connect()
            gripper.set_position(500)
            self.assertTrue(wait_for(
                lambda: any(f.hex().startswith("0106010301f4") for f in transport.written)),
                transport.written)
            self.assertEqual(gripper.position_raw, 250)
        finally:
            gripper.close()

    def test_position_clamped_to_range(self):
        gripper, transport = self._make([status_response()] * 50)
        try:
            gripper.connect()
            gripper.set_position(5000)
            self.assertTrue(wait_for(
                lambda: any(f.hex().startswith("0106010303e8") for f in transport.written)))
            gripper.set_position(-5)
            self.assertTrue(wait_for(
                lambda: any(f.hex().startswith("010601030000") for f in transport.written)))
        finally:
            gripper.close()

    def test_hold_cancels_pending_target(self):
        gripper, transport = self._make([status_response()] * 50)
        try:
            gripper.connect()
            gripper.set_position(800)
            gripper.hold()
            count_before = sum(1 for f in transport.written if f[2:4] == bytes([0x01, 0x03]))
            time.sleep(0.1)
            count_after = sum(1 for f in transport.written if f[2:4] == bytes([0x01, 0x03]))
            self.assertEqual(count_after, count_before, "hold 后不应再下发目标")
        finally:
            gripper.close()

    def test_connect_failure_raises_gripper_error(self):
        transport = FakeTransport([])   # read 返回空 → 初始化读状态失败
        gripper = Gripper(transport=transport)
        with self.assertRaises(GripperError):
            gripper.connect()

    def test_position_max_constant(self):
        self.assertEqual(POSITION_MAX, 1000)


if __name__ == "__main__":
    unittest.main()
