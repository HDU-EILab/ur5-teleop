"""Timestamp alignment and actual CFR video export regressions."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_lerobot import synchronized_frames, cut_video, segment_ranges
from ur5_teleop.frames import rotvec_to_matrix


class AlignmentTest(unittest.TestCase):
    def test_camera_stall_splits_continuous_segments(self):
        times = np.arange(180) / 30
        camera = np.concatenate([times[:3], times[16:]])
        timeline, indices = synchronized_frames(camera, times, times, 30)
        valid = np.abs(camera[indices[0]] - timeline) <= .05
        segments = segment_ranges(len(timeline), [], 0, timeline, .25, 45, valid)
        self.assertEqual(segments, [(15, 179)])
        # A later gap creates separate episodes, preserving the 30 Hz time axis.
        valid[80:90] = False
        self.assertEqual(segment_ranges(len(timeline), [], 0, timeline, .25, 45, valid),
                         [(15, 79), (90, 179)])

    def test_common_timeline_uses_different_camera_indices(self):
        global_times = np.arange(18) / 10
        wrist_times = global_times + .12
        robot_times = np.arange(201) / 100
        timeline, (g, w, r) = synchronized_frames(global_times, wrist_times, robot_times, 10)
        self.assertAlmostEqual(timeline[0], .12)
        self.assertEqual(g[0], 1)
        self.assertEqual(w[0], 0)
        self.assertLessEqual(np.max(abs(global_times[g] - wrist_times[w])), .021)
        self.assertLessEqual(np.max(abs(robot_times[r] - timeline)), .0051)
        for invalid in ([], [0, 0], [1, 0], [0, np.nan]):
            with self.assertRaises(ValueError):
                synchronized_frames(invalid, [0, 1], [0, 1], 10)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for real video validation')
    def test_real_export_preserves_selected_frames_and_rebuilds_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'episodes' / '20261111_test'
            (source / 'videos').mkdir(parents=True)
            for camera, offset in [('global', 0.), ('wrist', .12)]:
                writer = cv2.VideoWriter(str(source / 'videos' / f'{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
                self.assertTrue(writer.isOpened())
                for index in range(18):
                    writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
                writer.release()
                with (source / 'videos' / f'{camera}_timestamps.csv').open('w') as f:
                    writer_csv = csv.DictWriter(f, fieldnames=['t_mono'])
                    writer_csv.writeheader()
                    writer_csv.writerows({'t_mono': index / 10 + offset} for index in range(18))
            fields = ['t_mono', 'robot_x', 'robot_y', 'robot_z', 'robot_rx', 'robot_ry', 'robot_rz',
                      'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'gripper_position', 'target_x', 'target_y',
                      'target_z', 'target_rx', 'target_ry', 'target_rz', 'trigger_right']
            with (source / 'robot.csv').open('w') as f:
                writer_csv = csv.DictWriter(f, fieldnames=fields)
                writer_csv.writeheader()
                for index in range(201):
                    row = dict.fromkeys(fields, 0.)
                    row.update(t_mono=index / 100, robot_x=index / 1000, robot_ry=-3.14,
                               target_ry=-3.14, gripper_position=700, trigger_right=1)
                    writer_csv.writerow(row)
            (source / 'meta.json').write_text(json.dumps({'started_at': 0, 'disturbance_windows': []}))
            output = root / 'aligned'
            subprocess.run([sys.executable, str(ROOT / 'tools/export_lerobot.py'), '--episodes-dir', str(root / 'episodes'),
                            '--out', str(output), '--fps', '10', '--min-segment-s', '.2'], check=True, capture_output=True)
            frame = pd.read_parquet(output / 'data/chunk-000/episode_000000.parquet')
            info = json.loads((output / 'meta/info.json').read_text())
            self.assertEqual(info['total_frames'], len(frame))
            self.assertEqual(info['features']['observation.images.global']['shape'], [48, 64, 3])
            state = frame['observation.state'].iloc[0]
            self.assertAlmostEqual(state[0], .012, places=6)
            self.assertGreater(state[4], 0)
            np.testing.assert_allclose(rotvec_to_matrix(state[3:6]), rotvec_to_matrix([0, -3.14, 0]), atol=1e-6)
            for camera, expected_first in [('global', 10), ('wrist', 0)]:
                cap = cv2.VideoCapture(str(output / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frame))
                ok, image = cap.read()
                cap.release()
                self.assertTrue(ok)
                self.assertLess(abs(float(image.mean()) - expected_first), 5)
            # Reusing an output path must leave the completed data intact.
            again = subprocess.run([sys.executable, str(ROOT / 'tools/export_lerobot.py'), '--episodes-dir', str(root / 'episodes'), '--out', str(output)], capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertTrue((output / 'meta/info.json').exists())
            repeated = root / 'repeated.mp4'
            cut_video(source / 'videos/global.mp4', repeated, [1, 1, 3, 5], 10)
            cap = cv2.VideoCapture(str(repeated))
            means = []
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                means.append(image.mean())
            cap.release()
            self.assertEqual(len(means), 4)
            np.testing.assert_allclose(means, [10, 10, 30, 50], atol=5)

            with (source / 'videos/global_timestamps.csv').open('w') as f:
                writer_csv = csv.DictWriter(f, fieldnames=['t_mono'])
                writer_csv.writeheader()
                writer_csv.writerows({'t_mono': i / 10 + (0.5 if i >= 8 else 0)} for i in range(18))
            gap_output = root / 'with-gap'
            subprocess.run([sys.executable, str(ROOT / 'tools/export_lerobot.py'), '--episodes-dir', str(root / 'episodes'),
                            '--out', str(gap_output), '--fps', '10', '--min-segment-s', '.2'], check=True, capture_output=True)
            cuts = json.loads((gap_output / 'meta/cut_info.json').read_text())
            self.assertEqual(len(cuts), 2)
            self.assertLess(cuts[0]['frames'][1] + 1, cuts[1]['frames'][0])
            for cut in cuts:
                episode = cut['episode_index']
                frame = pd.read_parquet(gap_output / f'data/chunk-000/episode_{episode:06d}.parquet')
                np.testing.assert_allclose(frame['timestamp'], np.arange(len(frame)) / 10, atol=1e-6)
                for camera in ('global', 'wrist'):
                    cap = cv2.VideoCapture(str(gap_output / f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'))
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frame))
                    cap.release()
            alignment = json.loads((gap_output / 'meta/alignment.json').read_text())['sources'][0]
            self.assertGreater(alignment['invalid_timing_frames'], 0)
            self.assertLessEqual(alignment['retained_max_source_offset_ms'], 150)


if __name__ == '__main__':
    unittest.main()
