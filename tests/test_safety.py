"""``ur5_teleop.safety`` 单测：通信看门狗、偏差看门狗、状态机判定。"""

import math
import unittest

from ur5_teleop.safety import (CommStatus, CommWatchdog, DeviationWatchdog, TeleopState,
                               decide_state, pose_error)

IDENTITY_POSE = (0.4, 0.0, 0.4, 0.0, 0.0, 0.0)


class CommWatchdogTest(unittest.TestCase):
    def make_watchdog(self, input_timeout_s=0.2, state_timeout_s=0.1):
        return CommWatchdog(input_timeout_s, state_timeout_s)

    def test_all_fresh(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=9.9,
                                             last_state_receive=9.95)
        self.assertTrue(status.input_fresh)
        self.assertTrue(status.state_fresh)
        self.assertAlmostEqual(status.input_age_s, 0.1)
        self.assertAlmostEqual(status.state_age_s, 0.05)

    def test_boundary_is_fresh(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=9.8,
                                             last_state_receive=9.9)
        self.assertTrue(status.input_fresh)
        self.assertTrue(status.state_fresh)

    def test_input_timeout(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=9.7,
                                             last_state_receive=10.0)
        self.assertFalse(status.input_fresh)
        self.assertTrue(status.state_fresh)

    def test_state_timeout(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=10.0,
                                             last_state_receive=9.8)
        self.assertTrue(status.input_fresh)
        self.assertFalse(status.state_fresh)

    def test_never_received_has_none_age(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=None,
                                             last_state_receive=None)
        self.assertIsNone(status.input_age_s)
        self.assertIsNone(status.state_age_s)
        self.assertFalse(status.input_fresh)
        self.assertFalse(status.state_fresh)

    def test_future_timestamp_is_clamped_to_zero(self):
        status = self.make_watchdog().update(now=10.0, last_input_receive=10.5,
                                             last_state_receive=10.5)
        self.assertEqual(status.input_age_s, 0.0)
        self.assertEqual(status.state_age_s, 0.0)
        self.assertTrue(status.input_fresh)
        self.assertTrue(status.state_fresh)


class DecideStateTest(unittest.TestCase):
    FRESH = CommStatus(input_fresh=True, state_fresh=True,
                       input_age_s=0.01, state_age_s=0.01)
    NO_INPUT_YET = CommStatus(input_fresh=False, state_fresh=True,
                              input_age_s=None, state_age_s=0.01)
    INPUT_STALE = CommStatus(input_fresh=False, state_fresh=True,
                             input_age_s=0.5, state_age_s=0.01)
    STATE_STALE = CommStatus(input_fresh=True, state_fresh=False,
                             input_age_s=0.01, state_age_s=0.5)

    def test_fault_wins(self):
        self.assertIs(decide_state(comm=self.FRESH, tracking=True, fault=True),
                      TeleopState.FAULT)

    def test_state_stale_is_fault(self):
        self.assertIs(decide_state(comm=self.STATE_STALE, tracking=True, fault=False),
                      TeleopState.FAULT)

    def test_input_stale_is_hold(self):
        self.assertIs(decide_state(comm=self.INPUT_STALE, tracking=True, fault=False),
                      TeleopState.HOLD)

    def test_tracking_when_all_fresh(self):
        self.assertIs(decide_state(comm=self.FRESH, tracking=True, fault=False),
                      TeleopState.TRACKING)
        self.assertIs(decide_state(comm=self.FRESH, tracking=False, fault=False),
                      TeleopState.IDLE)

    def test_never_received_input_is_idle_not_hold(self):
        self.assertIs(decide_state(comm=self.NO_INPUT_YET, tracking=False, fault=False),
                      TeleopState.IDLE)


class PoseErrorTest(unittest.TestCase):
    def test_identical_poses(self):
        position_error, rotation_error = pose_error(IDENTITY_POSE, IDENTITY_POSE)
        self.assertAlmostEqual(position_error, 0.0)
        self.assertAlmostEqual(rotation_error, 0.0)

    def test_position_error(self):
        target = (0.5, 0.0, 0.4, 0.0, 0.0, 0.0)
        position_error, rotation_error = pose_error(target, IDENTITY_POSE)
        self.assertAlmostEqual(position_error, 0.1, places=9)
        self.assertAlmostEqual(rotation_error, 0.0, places=9)

    def test_rotation_error_quarter_turn(self):
        target = (0.4, 0.0, 0.4, 0.0, 0.0, math.pi / 2)
        position_error, rotation_error = pose_error(target, IDENTITY_POSE)
        self.assertAlmostEqual(position_error, 0.0, places=9)
        self.assertAlmostEqual(rotation_error, math.pi / 2, places=6)

    def test_rotation_error_half_turn(self):
        target = (0.4, 0.0, 0.4, 0.0, 0.0, math.pi)
        _position_error, rotation_error = pose_error(target, IDENTITY_POSE)
        self.assertAlmostEqual(rotation_error, math.pi, places=6)


class DeviationWatchdogTest(unittest.TestCase):
    def make_watchdog(self):
        return DeviationWatchdog(pos_threshold_m=0.05, rot_threshold_rad=0.26, hold_s=0.2)

    def test_within_threshold_never_faults(self):
        watchdog = self.make_watchdog()
        target = (0.42, 0.0, 0.4, 0.0, 0.0, 0.0)  # 偏差 0.02 < 0.05
        for step in range(5):
            self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=step * 0.1))
        self.assertFalse(watchdog.faulted)

    def test_exceeding_but_not_long_enough_does_not_fault(self):
        watchdog = self.make_watchdog()
        target = (0.6, 0.0, 0.4, 0.0, 0.0, 0.0)  # 偏差 0.2
        self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=0.0))
        self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=0.19))
        self.assertFalse(watchdog.faulted)

    def test_sustained_deviation_faults(self):
        watchdog = self.make_watchdog()
        target = (0.6, 0.0, 0.4, 0.0, 0.0, 0.0)
        self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=0.0))
        self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=0.1))
        self.assertTrue(watchdog.update(target, IDENTITY_POSE, now=0.2))
        self.assertTrue(watchdog.faulted)

    def test_recovery_before_hold_resets_timer(self):
        watchdog = self.make_watchdog()
        bad = (0.6, 0.0, 0.4, 0.0, 0.0, 0.0)
        good = (0.42, 0.0, 0.4, 0.0, 0.0, 0.0)
        self.assertFalse(watchdog.update(bad, IDENTITY_POSE, now=0.0))
        self.assertFalse(watchdog.update(good, IDENTITY_POSE, now=0.1))
        self.assertIsNone(watchdog.since)
        self.assertFalse(watchdog.update(bad, IDENTITY_POSE, now=0.2))
        self.assertFalse(watchdog.faulted)

    def test_rotation_deviation_faults(self):
        watchdog = self.make_watchdog()
        target = (0.4, 0.0, 0.4, 0.0, 0.0, 0.5)  # 0.5 rad > 0.26
        self.assertFalse(watchdog.update(target, IDENTITY_POSE, now=0.0))
        self.assertTrue(watchdog.update(target, IDENTITY_POSE, now=0.2))

    def test_fault_is_latched_until_reset(self):
        watchdog = self.make_watchdog()
        bad = (0.6, 0.0, 0.4, 0.0, 0.0, 0.0)
        watchdog.update(bad, IDENTITY_POSE, now=0.0)
        self.assertTrue(watchdog.update(bad, IDENTITY_POSE, now=0.2))
        # 即使回到阈值内也保持故障
        self.assertTrue(watchdog.update(IDENTITY_POSE, IDENTITY_POSE, now=0.3))
        self.assertTrue(watchdog.faulted)
        watchdog.reset()
        self.assertFalse(watchdog.faulted)
        self.assertIsNone(watchdog.since)
        self.assertFalse(watchdog.update(bad, IDENTITY_POSE, now=1.0))


if __name__ == "__main__":
    unittest.main()
