from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class InitialTargetReview:
    frame_bgr: np.ndarray
    frame_index: int
    width: int
    height: int
    fps: float
    candidate_points: list[tuple[int, int]]
    auto_point: tuple[int, int]
    user_seed_point: tuple[int, int] | None
    heatmap_peak: float


@dataclass
class TargetPointValidation:
    score: float
    level: str
    survivability_ratio: float
    mean_feature_count: float
    mean_retention_ratio: float
    motion_span_px: float
    reacquire_count: int
    lost_frame_count: int
    sample_summaries: list[dict]


class TargetPointProposer:
    def __init__(self, init_frames_limit=18, downsample_factor=4):
        self.init_frames_limit = max(8, int(init_frames_limit))
        self.downsample_factor = max(2, int(downsample_factor))
        self.min_candidate_spacing_ratio = 0.09

    def prepare(self, video_path, user_seed_ratio=None, limit=6):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        small_width = max(1, width // self.downsample_factor)
        small_height = max(1, height // self.downsample_factor)
        motion_heatmap = np.zeros((small_height, small_width), dtype=np.float32)

        representative_frame = None
        prev_gray = None
        frame_index = -1
        processed_frames = 0

        try:
            while processed_frames < self.init_frames_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                representative_frame = frame.copy()

                small_frame = cv2.resize(frame, (small_width, small_height))
                curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    motion_heatmap += self._build_motion_map(prev_gray, curr_gray)

                prev_gray = curr_gray
                processed_frames += 1
        finally:
            cap.release()

        if representative_frame is None:
            raise RuntimeError(f"Could not read frames from video: {video_path}")

        auto_point, heatmap_peak = self._pick_auto_point(motion_heatmap, width, height)
        user_seed_point = self._seed_to_absolute(user_seed_ratio, width, height)
        candidate_points = self._build_candidates(
            motion_heatmap=motion_heatmap,
            width=width,
            height=height,
            auto_point=auto_point,
            user_seed_point=user_seed_point,
            limit=limit,
        )

        return InitialTargetReview(
            frame_bgr=representative_frame,
            frame_index=frame_index,
            width=width,
            height=height,
            fps=fps,
            candidate_points=candidate_points,
            auto_point=auto_point,
            user_seed_point=user_seed_point,
            heatmap_peak=heatmap_peak,
        )

    def validate_point(self, video_path, point, start_frame_index=0, sample_count=6):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        point = self._clip_point(point, width, height)
        start_frame_index = max(0, min(max(0, total_frames - 1), int(start_frame_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Could not read validation start frame.")

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked_point = point
        features = self._seed_features(prev_gray, tracked_point, width, height)
        if features is not None:
            prev_feature_count = len(features)
        else:
            prev_feature_count = 0

        processed_frames = 1
        valid_frames = 1 if prev_feature_count >= 6 else 0
        feature_counts = [prev_feature_count]
        retention_ratios = [1.0 if prev_feature_count > 0 else 0.0]
        tracked_positions = [tracked_point]
        reacquire_count = 0
        lost_frame_count = 0

        frames_remaining = max(1, total_frames - start_frame_index)
        sample_targets = {
            max(0, min(total_frames - 1, int(start_frame_index + ratio * max(0, frames_remaining - 1))))
            for ratio in np.linspace(0.0, 1.0, num=max(2, int(sample_count)))
        }
        sample_summaries = []

        current_frame_index = start_frame_index

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_index += 1
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            retention_ratio = 0.0
            feature_count = 0
            status = "lost"

            if features is not None and len(features) >= 4:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = features[st == 1]
                    feature_count = len(good_new)
                    retention_ratio = feature_count / max(1, prev_feature_count)
                    if feature_count >= 4:
                        displacement = np.median(good_new - good_old, axis=0)
                        tracked_point = self._clip_point(
                            (
                                int(round(tracked_point[0] + float(displacement[0]))),
                                int(round(tracked_point[1] + float(displacement[1]))),
                            ),
                            width,
                            height,
                        )
                        features = good_new.reshape(-1, 1, 2)
                        prev_feature_count = feature_count
                        valid_frames += 1
                        status = "tracked"
                    else:
                        features = None

            if features is None or len(features) < 4:
                reacquired = self._seed_features(curr_gray, tracked_point, width, height)
                reacquire_count += 1
                if reacquired is not None and len(reacquired) >= 6:
                    features = reacquired
                    feature_count = len(reacquired)
                    prev_feature_count = feature_count
                    retention_ratio = max(retention_ratio, 0.35)
                    valid_frames += 1
                    status = "reacquired"
                else:
                    features = None
                    feature_count = 0
                    prev_feature_count = 0
                    lost_frame_count += 1
                    status = "lost"

            feature_counts.append(feature_count)
            retention_ratios.append(retention_ratio)
            tracked_positions.append(tracked_point)
            processed_frames += 1

            if current_frame_index in sample_targets:
                sample_summaries.append(
                    {
                        "frame": int(current_frame_index),
                        "point": tracked_point,
                        "feature_count": int(feature_count),
                        "retention_ratio": round(float(retention_ratio), 3),
                        "status": status,
                    }
                )

            prev_gray = curr_gray

        cap.release()

        tracked_y = [position[1] for position in tracked_positions]
        motion_span_px = float(max(tracked_y) - min(tracked_y)) if tracked_y else 0.0
        survivability_ratio = valid_frames / max(1, processed_frames)
        mean_feature_count = float(np.mean(feature_counts)) if feature_counts else 0.0
        mean_retention_ratio = float(np.mean(retention_ratios)) if retention_ratios else 0.0

        x_norm = point[0] / max(1.0, float(width))
        y_norm = point[1] / max(1.0, float(height))
        center_bias = max(0.0, 1.0 - (abs(x_norm - 0.5) / 0.5))
        lower_bias = max(0.0, 1.0 - (abs(y_norm - 0.62) / 0.42))
        location_score = (0.55 * center_bias) + (0.45 * lower_bias)

        score = (
            (0.38 * survivability_ratio)
            + (0.18 * min(1.0, mean_feature_count / 18.0))
            + (0.14 * min(1.0, mean_retention_ratio))
            + (0.10 * min(1.0, motion_span_px / 120.0))
            + (0.20 * location_score)
        )
        score -= min(0.24, reacquire_count / max(1, processed_frames) * 1.80)
        score -= min(0.24, lost_frame_count / max(1, processed_frames) * 0.60)
        if mean_feature_count < 8.0:
            score -= min(0.18, (8.0 - mean_feature_count) * 0.03)
        score = max(0.0, min(1.0, score))

        if score >= 0.74:
            level = "HIGH"
        elif score >= 0.48:
            level = "MEDIUM"
        else:
            level = "LOW"

        if not sample_summaries:
            sample_summaries.append(
                {
                    "frame": int(start_frame_index),
                    "point": tracked_point,
                    "feature_count": int(prev_feature_count),
                    "retention_ratio": 1.0 if prev_feature_count > 0 else 0.0,
                    "status": "tracked" if prev_feature_count > 0 else "lost",
                }
            )

        return TargetPointValidation(
            score=score,
            level=level,
            survivability_ratio=survivability_ratio,
            mean_feature_count=mean_feature_count,
            mean_retention_ratio=mean_retention_ratio,
            motion_span_px=motion_span_px,
            reacquire_count=reacquire_count,
            lost_frame_count=lost_frame_count,
            sample_summaries=sample_summaries[: max(2, sample_count)],
        )

    @staticmethod
    def _build_motion_map(prev_gray_small, curr_gray_small):
        flow = cv2.calcOpticalFlowFarneback(prev_gray_small, curr_gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.GaussianBlur(mag, (5, 5), 0)

    def _pick_auto_point(self, motion_heatmap, width, height):
        if motion_heatmap.size == 0 or float(np.max(motion_heatmap)) <= 1e-6:
            return (width // 2, height // 2), 0.0

        blurred = cv2.GaussianBlur(motion_heatmap, (5, 5), 0)
        _, peak, _, max_loc = cv2.minMaxLoc(blurred)
        auto_point = (
            int(max_loc[0] * self.downsample_factor),
            int(max_loc[1] * self.downsample_factor),
        )
        return self._clip_point(auto_point, width, height), float(peak)

    def _build_candidates(self, motion_heatmap, width, height, auto_point, user_seed_point, limit):
        scored_points = []
        if user_seed_point is not None:
            scored_points.append((user_seed_point, 1.35))

        scored_points.append((auto_point, 1.10))
        for point in self._auto_bias_points(auto_point, width, height):
            scored_points.append((point, self._candidate_score(point, motion_heatmap, width, height, auto_point, user_seed_point)))

        for point in self._motion_hotspots(motion_heatmap, width, height, limit=max(limit * 6, 24)):
            scored_points.append((point, self._candidate_score(point, motion_heatmap, width, height, auto_point, user_seed_point)))

        selected_points = []
        seen = set()
        min_spacing = max(56.0, min(width, height) * self.min_candidate_spacing_ratio)

        for point, score in sorted(scored_points, key=lambda item: item[1], reverse=True):
            point = self._clip_point(point, width, height)
            if point in seen:
                continue
            if any(self._point_distance(point, existing) < min_spacing for existing in selected_points):
                continue
            seen.add(point)
            selected_points.append(point)
            if len(selected_points) >= limit:
                break

        if auto_point not in selected_points and len(selected_points) < limit:
            selected_points.append(auto_point)

        if len(selected_points) < limit:
            for point in self._strategic_points(width, height):
                point = self._clip_point(point, width, height)
                if point in seen:
                    continue
                if any(self._point_distance(point, existing) < min_spacing for existing in selected_points):
                    continue
                seen.add(point)
                selected_points.append(point)
                if len(selected_points) >= limit:
                    break

        return selected_points[:limit]

    @staticmethod
    def _seed_to_absolute(seed_ratio, width, height):
        if seed_ratio is None:
            return None

        x = int(float(seed_ratio[0]) * width)
        y = int(float(seed_ratio[1]) * height)
        return (
            max(0, min(width - 1, x)),
            max(0, min(height - 1, y)),
        )

    @staticmethod
    def _clip_point(point, width, height):
        x = int(max(0, min(width - 1, point[0])))
        y = int(max(0, min(height - 1, point[1])))
        return (x, y)

    @staticmethod
    def _square_mask(width, height, point, half_size=40):
        mask = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, int(point[0] - half_size))
        y0 = max(0, int(point[1] - half_size))
        x1 = min(width, int(point[0] + half_size))
        y1 = min(height, int(point[1] + half_size))
        mask[y0:y1, x0:x1] = 255
        return mask

    def _seed_features(self, gray_frame, point, width, height):
        mask = self._square_mask(width, height, point)
        return cv2.goodFeaturesToTrack(
            gray_frame,
            mask=mask,
            maxCorners=40,
            qualityLevel=0.08,
            minDistance=5,
        )

    @staticmethod
    def _auto_bias_points(auto_point, width, height):
        step_x = max(48, int(width * 0.055))
        step_y = max(42, int(height * 0.065))
        x, y = auto_point
        return [
            (x, y + step_y),
            (x - step_x, y + step_y),
            (x + step_x, y + step_y),
            (x - step_x, y),
            (x + step_x, y),
            (x, y - step_y),
        ]

    def _candidate_score(self, point, motion_heatmap, width, height, auto_point, user_seed_point):
        point = self._clip_point(point, width, height)
        motion_score = self._motion_score(point, motion_heatmap)

        x_norm = point[0] / max(1.0, float(width))
        y_norm = point[1] / max(1.0, float(height))
        center_bias = max(0.0, 1.0 - (abs(x_norm - 0.5) / 0.5))
        lower_bias = max(0.0, 1.0 - (abs(y_norm - 0.62) / 0.42))

        auto_bias = 0.0
        if auto_point is not None:
            auto_dist = self._point_distance(point, auto_point)
            auto_bias = max(0.0, 1.0 - (auto_dist / max(96.0, min(width, height) * 0.18)))

        seed_bias = 0.0
        if user_seed_point is not None:
            seed_dist = self._point_distance(point, user_seed_point)
            seed_bias = max(0.0, 1.0 - (seed_dist / max(90.0, min(width, height) * 0.16)))

        return (
            (0.46 * motion_score)
            + (0.20 * center_bias)
            + (0.18 * lower_bias)
            + (0.10 * auto_bias)
            + (0.06 * seed_bias)
        )

    def _motion_score(self, point, motion_heatmap):
        if motion_heatmap is None or motion_heatmap.size == 0:
            return 0.35

        h, w = motion_heatmap.shape[:2]
        x = int(max(0, min(w - 1, round(point[0] / float(self.downsample_factor)))))
        y = int(max(0, min(h - 1, round(point[1] / float(self.downsample_factor)))))
        value = float(motion_heatmap[y, x])
        peak = float(np.max(motion_heatmap))
        if peak <= 1e-6:
            return 0.35
        return max(0.0, min(1.0, value / peak))

    @staticmethod
    def _point_distance(point_a, point_b):
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    def _motion_hotspots(self, motion_heatmap, width, height, limit):
        if motion_heatmap is None or motion_heatmap.size == 0:
            return []

        blurred = cv2.GaussianBlur(motion_heatmap, (5, 5), 0)
        flat = blurred.flatten()
        candidate_count = min(limit, flat.size)
        if candidate_count <= 0:
            return []

        indices = np.argpartition(flat, -candidate_count)[-candidate_count:]
        ranked = sorted(indices, key=lambda idx: float(flat[idx]), reverse=True)

        hotspots = []
        for flat_index in ranked:
            y, x = np.unravel_index(flat_index, blurred.shape)
            hotspots.append(
                self._clip_point(
                    (int(x * self.downsample_factor), int(y * self.downsample_factor)),
                    width,
                    height,
                )
            )
        return hotspots

    @staticmethod
    def _strategic_points(width, height):
        return [
            (int(width * 0.50), int(height * 0.62)),
            (int(width * 0.42), int(height * 0.66)),
            (int(width * 0.58), int(height * 0.66)),
            (int(width * 0.34), int(height * 0.60)),
            (int(width * 0.66), int(height * 0.60)),
        ]
