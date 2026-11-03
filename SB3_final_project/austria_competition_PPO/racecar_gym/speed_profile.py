"""Curvature based speed profile of a track, computed offline from the track maps.

The profile gives, for every progress value in [0, 1), the maximum speed that still allows the car to
slow down in time for the sharpest corner ahead (a "braking envelope"). It is used only to shape the
training reward (privileged information); the policy itself still sees only the camera image.
"""
import os
from typing import List

import numpy as np

DEFAULT_MAPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models', 'scenes', 'austria_competition', 'maps', 'maps.npz')


def _circular_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x
    xp = np.concatenate([x[-k:], x, x[:k]], axis=0)
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    if x.ndim == 1:
        return np.convolve(xp, kernel, mode='valid')
    return np.stack([np.convolve(xp[:, i], kernel, mode='valid') for i in range(x.shape[1])], axis=1)


class SpeedProfile:
    def __init__(self,
                 maps_path: str = DEFAULT_MAPS,
                 resolution: float = 0.05,
                 n_bins: int = 400,
                 v_max: float = 5.0,
                 v_min: float = 1.0,
                 a_lat: float = 4.0,
                 a_brake: float = 3.0,
                 curvature_smooth_bins: int = 6):
        self.n_bins = n_bins
        self.v_max = v_max
        maps = np.load(maps_path, allow_pickle=True)
        progress_map = maps['norm_distance_from_start']
        obstacle_map = maps['norm_distance_to_obstacle']
        drivable = maps['drivable_area']

        # 1. Centerline: for every progress bin, the wall-distance weighted mean of the drivable pixels.
        points = []
        for i in range(n_bins):
            sel = (progress_map >= i / n_bins) & (progress_map < (i + 1) / n_bins) & (drivable > 0)
            idx = np.argwhere(sel)
            if len(idx) == 0:
                points.append(points[-1])
                continue
            w = obstacle_map[sel] ** 4 + 1e-9
            points.append((idx * w[:, None]).sum(0) / w.sum())
        points = _circular_smooth(np.array(points) * resolution, 5)

        # 2. Heading change per meter (curvature), smoothed over about 1.5 m.
        seg = np.roll(points, -1, axis=0) - points
        seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        dh = np.roll(heading, -1) - heading
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        curvature = _circular_smooth(np.abs(dh) / seg_len, curvature_smooth_bins)

        self.seg_len = seg_len
        self.lap_length = float(seg_len.sum())
        self.curvature = curvature

        # 3. Corner speed limit from the lateral acceleration budget.
        v_corner = np.clip(np.sqrt(a_lat / np.maximum(curvature, 1e-3)), v_min, v_max)

        # 4. Braking envelope: walk backwards around the (circular) lap, twice to close the loop.
        v_allowed = v_corner.copy()
        for _ in range(2):
            for i in range(n_bins - 1, -1, -1):
                nxt = v_allowed[(i + 1) % n_bins]
                v_allowed[i] = min(v_allowed[i], np.sqrt(nxt ** 2 + 2.0 * a_brake * seg_len[i]))
        self.v_corner = v_corner
        self.v_allowed = v_allowed

    def allowed_speed(self, progress: float) -> float:
        return float(self.v_allowed[int(progress * self.n_bins) % self.n_bins])

    def corner_entries(self, threshold_ratio: float = 0.7) -> List[float]:
        """Progress values where a corner (speed limit below threshold_ratio * v_max) begins."""
        slow = self.v_corner < threshold_ratio * self.v_max
        entries = []
        for i in range(self.n_bins):
            if slow[i] and not slow[i - 1]:
                entries.append(i / self.n_bins)
        return entries


if __name__ == '__main__':
    profile = SpeedProfile()
    print(f'lap length: {profile.lap_length:.1f} m')
    print('corner entries (progress):', [round(e, 3) for e in profile.corner_entries()])
    for i in range(0, profile.n_bins, 10):
        print(f'progress {i / profile.n_bins:.3f}  corner {profile.v_corner[i]:.2f}  allowed {profile.v_allowed[i]:.2f}')
