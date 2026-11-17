"""Discard camera startup buffers until exposure and colour have settled."""

from collections import deque

import cv2
import numpy as np


class FrameReadiness:
    def __init__(self, started_at, *, warmup_s=1.0, stable_frames=15):
        self.started_at = started_at
        self.warmup_s = warmup_s
        self.samples = deque(maxlen=stable_frames)
        self.ready = False
        self.discarded = 0

    def accept(self, frame, timestamp):
        if self.ready:
            return True
        pixels = cv2.resize(frame, (64, 36)).astype(np.float32)
        mean = pixels.mean(axis=(0, 1))
        # Blank buffers, including UVC's uniform green startup image, lack texture.
        if float(pixels.std(axis=(0, 1)).max()) < 2:
            self.samples.clear()
        else:
            self.samples.append((timestamp, mean))
        if len(self.samples) == self.samples.maxlen:
            span = timestamp - self.samples[0][0]
            drift = np.ptp(np.stack([mean for _, mean in self.samples]), axis=0)
            self.ready = (
                timestamp - self.started_at >= self.warmup_s
                and span >= 0.4
                and bool(np.all(drift <= 2))
            )
        if not self.ready:
            self.discarded += 1
        return self.ready
