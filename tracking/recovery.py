from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class RecoveryResult:
    mode: str
    point: tuple
    score: float
    appearance_similarity: float
    motion_score: float
    geometry_score: float
    mask_area_ratio: float
    mask: np.ndarray
    signature: dict
    feature_points: np.ndarray | None
    candidate_count: int


class TrackingRecoveryEngine:
    def __init__(self):
        self.local_offsets = [
            (0, 0),
            (-24, 0),
            (24, 0),
            (0, -24),
            (0, 24),
            (-36, -24),
            (36, -24),
            (-36, 24),
            (36, 24),
            (-56, 0),
            (56, 0),
        ]
        self.global_grid_cols = 5
        self.global_grid_rows = 4
        self.min_mask_area_ratio = 0.0010
        self.max_mask_area_ratio = 0.35

    def attempt_recovery(
        self,
        mode,
        frame,
        gray_frame,
        width,
        height,
        anchor_point,
        sam_handler,
        dino_handler,
        appearance_memory,
        motion_map=None,
    ):
        if sam_handler is None:
            return None

        candidates = self._build_candidates(mode, width, height, anchor_point, motion_map)
        if not candidates:
            return None

        best = None
        for point in candidates:
            point = self._clip_point(point, width, height)
            mask = sam_handler.get_mask(frame, point)
            if mask is None:
                continue

            mask_area_ratio = float(np.count_nonzero(mask)) / float(width * height)
            if mask_area_ratio < self.min_mask_area_ratio or mask_area_ratio > self.max_mask_area_ratio:
                continue

            signature = None
            if dino_handler is not None:
                signature = dino_handler.extract_features(frame, mask=mask, prompt_point=point)
            if signature is None:
                signature = {"centroid": point, "confidence": 0.0, "embedding": None, "point": point}
            signature.setdefault("point", point)

            appearance_similarity = 0.0
            if appearance_memory is not None and appearance_memory.has_reference():
                appearance_similarity = appearance_memory.similarity(signature.get("embedding"))
            else:
                appearance_similarity = float(signature.get("confidence", 0.0))

            motion_score = self._score_motion(point, motion_map)
            geometry_score = self._score_geometry(mode, point, anchor_point, width, height)
            mask_score = self._score_mask(mask_area_ratio)

            total_score = (
                (0.46 * appearance_similarity)
                + (0.24 * motion_score)
                + (0.18 * geometry_score)
                + (0.12 * mask_score)
            )

            feature_points = cv2.goodFeaturesToTrack(
                gray_frame,
                mask=mask,
                maxCorners=50,
                qualityLevel=0.1,
                minDistance=5,
            )

            candidate_result = RecoveryResult(
                mode=mode,
                point=signature.get("centroid") or point,
                score=float(total_score),
                appearance_similarity=float(appearance_similarity),
                motion_score=float(motion_score),
                geometry_score=float(geometry_score),
                mask_area_ratio=float(mask_area_ratio),
                mask=mask,
                signature=signature,
                feature_points=feature_points,
                candidate_count=len(candidates),
            )

            if best is None or candidate_result.score > best.score:
                best = candidate_result

        if best is None:
            return None

        threshold = 0.42 if mode == "local" else 0.47
        if appearance_memory is None or not appearance_memory.has_reference():
            threshold -= 0.10

        if best.score < threshold:
            return None

        return best

    def suggest_points(self, width, height, anchor_point=None, motion_map=None, limit=6):
        candidates = self._build_candidates("global", width, height, anchor_point, motion_map)
        if limit is not None:
            return candidates[:limit]
        return candidates

    def _build_candidates(self, mode, width, height, anchor_point, motion_map):
        points = []
        if mode == "local" and anchor_point is not None:
            for dx, dy in self.local_offsets:
                points.append((anchor_point[0] + dx, anchor_point[1] + dy))
            points.extend(self._motion_hotspots(motion_map, width, height, limit=4, local_anchor=anchor_point, local_radius=88))
        else:
            x_step = width / (self.global_grid_cols + 1)
            y_step = height / (self.global_grid_rows + 1)
            for row in range(1, self.global_grid_rows + 1):
                for col in range(1, self.global_grid_cols + 1):
                    points.append((int(col * x_step), int(row * y_step)))

            if anchor_point is not None:
                points.extend(
                    [
                        anchor_point,
                        (anchor_point[0] - 80, anchor_point[1]),
                        (anchor_point[0] + 80, anchor_point[1]),
                        (anchor_point[0], anchor_point[1] - 80),
                        (anchor_point[0], anchor_point[1] + 80),
                    ]
                )

            points.extend(self._motion_hotspots(motion_map, width, height, limit=8))

        unique_points = []
        seen = set()
        for point in points:
            point = self._clip_point(point, width, height)
            key = (int(point[0]), int(point[1]))
            if key not in seen:
                seen.add(key)
                unique_points.append(key)
        return unique_points

    @staticmethod
    def _clip_point(point, width, height):
        x = int(max(0, min(width - 1, point[0])))
        y = int(max(0, min(height - 1, point[1])))
        return (x, y)

    @staticmethod
    def _score_mask(mask_area_ratio):
        target = 0.03
        spread = 0.10
        delta = abs(mask_area_ratio - target)
        return max(0.0, 1.0 - (delta / spread))

    @staticmethod
    def _score_geometry(mode, point, anchor_point, width, height):
        if mode != "local" or anchor_point is None:
            center = (width / 2.0, height / 2.0)
            dist = math.dist(point, center)
            max_dist = math.dist((0, 0), center)
            return max(0.2, 1.0 - (dist / max_dist))

        dist = math.dist(point, anchor_point)
        return max(0.0, 1.0 - (dist / 96.0))

    @staticmethod
    def _score_motion(point, motion_map):
        if motion_map is None or motion_map.size == 0:
            return 0.35

        h, w = motion_map.shape[:2]
        x = int(max(0, min(w - 1, round(point[0] / 4.0))))
        y = int(max(0, min(h - 1, round(point[1] / 4.0))))
        value = float(motion_map[y, x])
        peak = float(np.max(motion_map))
        if peak <= 1e-6:
            return 0.35
        return max(0.0, min(1.0, value / peak))

    @staticmethod
    def _motion_hotspots(motion_map, width, height, limit=4, local_anchor=None, local_radius=None):
        if motion_map is None or motion_map.size == 0:
            return []

        blurred = cv2.GaussianBlur(motion_map, (5, 5), 0)
        flat_indices = np.argpartition(blurred.flatten(), -limit)[-limit:]
        hotspots = []
        for flat_index in flat_indices:
            y, x = np.unravel_index(flat_index, blurred.shape)
            point = (int(x * 4), int(y * 4))
            if local_anchor is not None and local_radius is not None:
                if math.dist(point, local_anchor) > local_radius:
                    continue
            hotspots.append(point)
        return hotspots
