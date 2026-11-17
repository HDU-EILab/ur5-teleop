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
        chunk = self.responses[0][:size]
        self.responses[0] = self.responses[0][size:]
        if not self.responses[0]:
            self.responses.pop(0)
        return chunk

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


class RtuResponseTest(unittest.TestCase):
    def test_partial_reads_are_assembled(self):
        frame = status_response(position=630)
        transport = FakeTransport([frame[:1], frame[1:4], frame[4:]])
        gripper = Gripper(transport=transport)
        gripper._read_status()
        self.assertEqual(gripper.position_raw, 630)
        self.assertTrue(gripper.healthy)

    def test_garbage_exception_bits_do_not_become_device_exceptions(self):
        transport = FakeTransport([bytes.fromhex('ffffffffff') + status_response(position=570)])
        gripper = Gripper(transport=transport)
        with self.assertLogs('ur5_teleop.gripper', level='WARNING'):
            gripper._read_status()
        self.assertEqual(gripper.position_raw, 570)
        self.assertEqual(gripper.communication_resyncs, 1)

    def test_crc_corruption_can_resynchronize_to_a_complete_valid_frame(self):
        damaged = bytearray(status_response(position=150))
        damaged[-1] ^= 1
        transport = FakeTransport([bytes(damaged) + status_response(position=250)])
        gripper = Gripper(transport=transport)
        with self.assertLogs('ur5_teleop.gripper', level='WARNING'):
            gripper._read_status()
        self.assertEqual(gripper.position_raw,250)
        self.assertEqual(len(transport.written),1)

    def test_corrupted_transaction_retries_and_accepts_only_a_valid_frame(self):
        class RetryTransport(FakeTransport):
            def write(self, frame):
                self.written.append(frame)
                if len(self.written) == 1:
                    self.responses.append(bytes.fromhex('5fa020414040207ab5d3'))
                else:
                    self.responses.append(status_response(position=1000))
        transport = RetryTransport()
        gripper = Gripper(transport=transport)
        gripper._response_timeout_s = .005
        with self.assertLogs('ur5_teleop.gripper', level='WARNING') as logs:
            gripper._read_status()
        self.assertEqual(len(transport.written), 2)
        self.assertEqual(gripper.communication_retries, 1)
        self.assertEqual(gripper.position_raw, 1000)
        self.assertTrue(gripper.healthy)
        self.assertIn('retry 1/1', logs.output[0])

    def test_repeated_invalid_transactions_fail_without_publishing_a_position(self):
        transport = FakeTransport()
        gripper = Gripper(transport=transport)
        gripper._response_timeout_s = .005
        with self.assertLogs('ur5_teleop.gripper', level='WARNING'):
            with self.assertRaisesRegex(GripperError, 'timeout/invalid frame'):
                gripper._read_status()
        self.assertEqual(len(transport.written), 2)
        self.assertIsNone(gripper.position_raw)
        self.assertFalse(gripper.healthy)

    def test_valid_five_byte_exception_is_reported_with_raw_frame(self):
        payload = bytes.fromhex('018302')
        frame = payload + crc16(payload).to_bytes(2,'little')
        gripper = Gripper(transport=FakeTransport([frame]))
        with self.assertRaisesRegex(GripperError,'0x02; frame=018302'):
            gripper._read_status()
        self.assertEqual(len(gripper._transport.written), 1)

    def test_invalid_exception_crc_is_not_reported_as_a_device_exception(self):
        gripper = Gripper(transport=FakeTransport([bytes.fromhex('0183ff0000')]))
        gripper._response_timeout_s = .01
        with self.assertRaisesRegex(GripperError,'timeout/invalid frame'):
            gripper._read_status()
        self.assertIsNone(gripper.position_raw)

    def test_response_from_another_slave_is_not_accepted(self):
        gripper = Gripper(transport=FakeTransport([status_response(slave_id=2)]))
        gripper._response_timeout_s = .01
        with self.assertRaisesRegex(GripperError,'timeout/invalid frame'):
            gripper._read_status()

    def test_write_echo_must_match_the_requested_register_and_value(self):
        class WrongEcho(FakeTransport):
            def write(self,frame):
                self.written.append(frame)
                self.responses.append(build_write(1,0x0104,[50]))
        gripper = Gripper(transport=WrongEcho())
        with self.assertRaisesRegex(GripperError,'回显不匹配'):
            gripper._write_register(0x0103,1000)

    def test_frame_gap_is_observed_between_transactions(self):
        transport = FakeTransport([status_response(),status_response()])
        gripper = Gripper(transport=transport)
        gripper._read_status()
        before = time.monotonic()
        gripper._read_status()
        self.assertGreaterEqual(time.monotonic()-before,gripper._frame_gap_s*.8)


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
