from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .funscript_generator import FunscriptSignalSample, PeakValleyFunscriptGenerator
from .models import SignalProvenance, TrackingState


@dataclass
class TwoPointTrackResult:
    data: dict
    diagnostics: list[dict]


class TwoPointTracker:
    def __init__(self, callback=None):
        self.callback = callback
        self.recent_positions = deque(maxlen=48)
        self.bridge_template = []
        self.bridge_step = 0
        self.lost_streak = 0
        self.pending_reacquire = None

    @staticmethod
    def _clip_point(point, width, height):
        return (
            int(max(0, min(width - 1, round(point[0])))),
            int(max(0, min(height - 1, round(point[1])))),
        )

    @staticmethod
    def _point_distance(a, b):
        return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    def _smooth_candidate(self, current, candidate, width, height, alpha, max_step):
        dx = float(candidate[0]) - float(current[0])
        dy = float(candidate[1]) - float(current[1])
        distance = float(np.hypot(dx, dy))
        if distance <= 1e-6:
            return current

        if distance > max_step:
            scale = max_step / distance
            candidate = (float(current[0]) + (dx * scale), float(current[1]) + (dy * scale))

        smoothed = (
            float(current[0]) + ((float(candidate[0]) - float(current[0])) * alpha),
            float(current[1]) + ((float(candidate[1]) - float(current[1])) * alpha),
        )
        return self._clip_point(smoothed, width, height)

    @staticmethod
    def _adaptive_acceptance(startup_lock, pair_score, p_template_score, r_template_score, width, height):
        min_dimension = float(min(width, height))
        if startup_lock:
            max_step = max(6.0, min(12.0, min_dimension * 0.006))
            alpha = 0.34
        else:
            max_step = max(12.0, min(30.0, min_dimension * 0.014))
            alpha = 0.52

        template_score = min(float(p_template_score), float(r_template_score))
        if pair_score < 0.72 or template_score < 0.62:
            max_step *= 0.68
            alpha *= 0.78
        elif pair_score >= 0.88 and template_score >= 0.82 and not startup_lock:
            max_step *= 1.18
            alpha *= 1.08

        return float(alpha), float(max_step)

    @staticmethod
    def _inside_frame(point, width, height, margin=0):
        return (
            margin <= float(point[0]) <= width - 1 - margin
            and margin <= float(point[1]) <= height - 1 - margin
        )

    @staticmethod
    def _is_black_frame(frame, mean_threshold=7.0, std_threshold=5.0):
        if frame is None or frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) <= mean_threshold and float(np.std(gray)) <= std_threshold

    @staticmethod
    def _feature_mask(width, height, point, radius):
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y = point
        x0 = max(0, int(x - radius))
        y0 = max(0, int(y - radius))
        x1 = min(width, int(x + radius))
        y1 = min(height, int(y + radius))
        mask[y0:y1, x0:x1] = 255
        return mask

    def _seed_features(self, gray, point, width, height, radius):
        mask = self._feature_mask(width, height, point, radius)
        return cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=45,
            qualityLevel=0.08,
            minDistance=5,
            blockSize=5,
        )

    @staticmethod
    def _extract_template(gray, point, radius, allow_flat=False):
        height, width = gray.shape[:2]
        x, y = point
        x0 = max(0, int(x - radius))
        y0 = max(0, int(y - radius))
        x1 = min(width, int(x + radius + 1))
        y1 = min(height, int(y + radius + 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        patch = gray[y0:y1, x0:x1].copy()
        if not allow_flat and float(np.std(patch)) < 3.0:
            return None
        return patch

    @staticmethod
    def _normalize_patch(patch):
        patch = patch.astype(np.float32)
        std = float(np.std(patch))
        if std < 1e-6:
            return None
        return (patch - float(np.mean(patch))) / std

    def _extract_pair_template(self, gray, primary, reference, output_size=96):
        height, width = gray.shape[:2]
        distance = self._point_distance(primary, reference)
        margin = int(max(42.0, min(width, height) * 0.035, distance * 0.32))
        x0 = max(0, int(min(primary[0], reference[0]) - margin))
        y0 = max(0, int(min(primary[1], reference[1]) - margin))
        x1 = min(width, int(max(primary[0], reference[0]) + margin + 1))
        y1 = min(height, int(max(primary[1], reference[1]) + margin + 1))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None

        patch = gray[y0:y1, x0:x1]
        if float(np.std(patch)) < 4.0:
            return None
        patch = cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return self._normalize_patch(patch)

    def _pair_template_score(self, gray, primary, reference, pair_template):
        if pair_template is None:
            return 0.0
        patch = self._extract_pair_template(gray, primary, reference, output_size=pair_template.shape[0])
        if patch is None:
            return 0.0
        score = float(np.mean(pair_template * patch))
        return max(0.0, min(1.0, (score + 1.0) * 0.5))

    def _update_pending_reacquire(self, candidate, max_drift):
        if candidate is None:
            self.pending_reacquire = None
            return 0

        if self.pending_reacquire is None:
            self.pending_reacquire = {
                "primary": candidate["primary"],
                "reference": candidate["reference"],
                "count": 1,
            }
            return 1

        primary_drift = self._point_distance(self.pending_reacquire["primary"], candidate["primary"])
        reference_drift = self._point_distance(self.pending_reacquire["reference"], candidate["reference"])
        if primary_drift <= max_drift and reference_drift <= max_drift:
            self.pending_reacquire = {
                "primary": candidate["primary"],
                "reference": candidate["reference"],
                "count": int(self.pending_reacquire["count"]) + 1,
            }
            return int(self.pending_reacquire["count"])

        self.pending_reacquire = {
            "primary": candidate["primary"],
            "reference": candidate["reference"],
            "count": 1,
        }
        return 1

    def _track_template(self, curr_gray, template, point, width, height, search_radius, min_score=0.52):
        if template is None:
            return point, 0.0, False, "no_template"

        t_h, t_w = template.shape[:2]
        half_w = t_w // 2
        half_h = t_h // 2
        x, y = point
        x0 = max(0, int(x - search_radius - half_w))
        y0 = max(0, int(y - search_radius - half_h))
        x1 = min(width, int(x + search_radius + half_w + 1))
        y1 = min(height, int(y + search_radius + half_h + 1))
        search = curr_gray[y0:y1, x0:x1]
        if search.shape[0] < t_h or search.shape[1] < t_w:
            return point, 0.0, False, "template_out_of_frame"

        template_std = float(np.std(template))
        search_std = float(np.std(search))
        if template_std < 3.0 or search_std < 3.0:
            result = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(result)
            score = 1.0 - float(min_val)
            max_loc = min_loc
        else:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            score = float(max_val)

        if score < min_score:
            return point, score, False, "template_low_score"

        new_x = x0 + max_loc[0] + half_w
        new_y = y0 + max_loc[1] + half_h
        if not self._inside_frame((new_x, new_y), width, height, margin=3):
            return point, score, False, "out_of_frame"
        return self._clip_point((new_x, new_y), width, height), score, True, "template"

    def _track_locked_point(self, curr_gray, template, point, width, height, search_radius):
        return self._track_template(
            curr_gray=curr_gray,
            template=template,
            point=point,
            width=width,
            height=height,
            search_radius=search_radius,
            min_score=0.48,
        )

    def _track_template_global(self, curr_gray, template, center_point, width, height, search_radius, min_score=0.62):
        return self._track_template(
            curr_gray=curr_gray,
            template=template,
            point=center_point,
            width=width,
            height=height,
            search_radius=search_radius,
            min_score=min_score,
        )

    def _verify_template_near_point(self, curr_gray, template, point, width, height, radius):
        _, score, _, reason = self._track_template(
            curr_gray=curr_gray,
            template=template,
            point=point,
            width=width,
            height=height,
            search_radius=radius,
            min_score=-1.0,
        )
        if reason in {"no_template", "template_out_of_frame", "out_of_frame"}:
            return 0.0
        return float(score)

    def _track_cloud(self, prev_gray, curr_gray, features, point, width, height, max_step):
        if features is None or len(features) < 4:
            return point, None, 0, 0.0, False, "no_features"

        prev_count = len(features)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
        if p1 is None or st is None:
            return point, None, 0, 0.0, False, "flow_failed"

        good_new = p1[st == 1]
        good_old = features[st == 1]
        count = len(good_new)
        retention = count / max(1, prev_count)
        if count < 4:
            return point, None, count, retention, False, "too_few_features"

        displacement = np.median(good_new - good_old, axis=0)
        step = float(np.hypot(float(displacement[0]), float(displacement[1])))
        if step > max_step:
            return point, None, count, retention, False, "large_step"

        predicted = (point[0] + float(displacement[0]), point[1] + float(displacement[1]))
        if not self._inside_frame(predicted, width, height, margin=3):
            return point, None, count, retention, False, "out_of_frame"

        new_point = self._clip_point(predicted, width, height)
        return new_point, good_new.reshape(-1, 1, 2), count, retention, True, "tracked"

    @staticmethod
    def _pair_is_valid(primary, reference, initial_distance, width, height):
        distance = float(np.hypot(primary[0] - reference[0], primary[1] - reference[1]))
        min_distance = max(28.0, min(width, height) * 0.025)
        if distance < min_distance:
            return False, "points_collapsed"
        if initial_distance > 1e-6:
            ratio = distance / initial_distance
            if ratio < 0.38 or ratio > 1.95:
                return False, "distance_ratio"
        return True, "ok"

    @staticmethod
    def _pair_geometry_score(primary, reference, initial_distance):
        distance = float(np.hypot(primary[0] - reference[0], primary[1] - reference[1]))
        if initial_distance <= 1e-6:
            return 0.0
        ratio = distance / initial_distance
        if ratio < 0.38 or ratio > 1.95:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (abs(ratio - 1.0) / 0.95)))

    @staticmethod
    def _pair_score(p_retention, r_retention, p_template_score, r_template_score, geometry_score):
        template_score = min(float(p_template_score), float(r_template_score))
        retention_score = min(float(p_retention), float(r_retention))
        return (
            (0.42 * max(0.0, min(1.0, template_score)))
            + (0.34 * max(0.0, min(1.0, retention_score)))
            + (0.24 * max(0.0, min(1.0, geometry_score)))
        )

    @staticmethod
    def _near_boundary(point, width, height):
        margin = max(48.0, min(width, height) * 0.065)
        return (
            point[0] <= margin
            or point[0] >= width - 1 - margin
            or point[1] <= margin
            or point[1] >= height - 1 - margin
        )

    def _try_reacquire_initial(
        self,
        curr_gray,
        width,
        height,
        initial_primary,
        initial_reference,
        initial_primary_template,
        initial_reference_template,
        initial_pair_template,
        initial_distance,
        radius,
        search_radius,
    ):
        primary, p_score, p_ok, p_reason = self._track_template_global(
            curr_gray, initial_primary_template, initial_primary, width, height, search_radius
        )
        reference, r_score, r_ok, r_reason = self._track_template_global(
            curr_gray, initial_reference_template, initial_reference, width, height, search_radius
        )
        if not p_ok or not r_ok:
            return None, p_score, r_score, p_reason if not p_ok else r_reason

        pair_ok, pair_reason = self._pair_is_valid(primary, reference, initial_distance, width, height)
        if not pair_ok:
            return None, p_score, r_score, pair_reason

        if self._near_boundary(primary, width, height) or self._near_boundary(reference, width, height):
            return None, p_score, r_score, "reacquire_boundary"

        pair_context_score = self._pair_template_score(curr_gray, primary, reference, initial_pair_template)
        if pair_context_score < 0.46:
            return None, p_score, r_score, "reacquire_context_low"

        primary_features = self._seed_features(curr_gray, primary, width, height, radius)
        reference_features = self._seed_features(curr_gray, reference, width, height, radius)
        if primary_features is None or reference_features is None or len(primary_features) < 6 or len(reference_features) < 6:
            return None, p_score, r_score, "reacquire_no_features"

        return {
            "primary": primary,
            "reference": reference,
            "primary_features": primary_features,
            "reference_features": reference_features,
            "pair_context_score": pair_context_score,
        }, p_score, r_score, "reacquired_initial"

    def _seed_bridge(self):
        if len(self.recent_positions) < 8:
            self.bridge_template = []
            self.bridge_step = 0
            return
        self.bridge_template = list(self.recent_positions)[-min(24, len(self.recent_positions)) :]
        self.bridge_step = 0

    def _next_bridge_pos(self, fallback=50, lost_streak=0):
        if not self.bridge_template:
            return int(fallback)
        value = self.bridge_template[self.bridge_step % len(self.bridge_template)]
        cycle = self.bridge_step // len(self.bridge_template)
        cycle_decay = max(0.62, 1.0 - (0.12 * cycle))
        lost_decay = max(0.42, 1.0 - (0.035 * max(0, int(lost_streak) - 1)))
        decay = min(cycle_decay, lost_decay)
        amplitude = max(-34.0, min(34.0, (float(value) - 50.0) * decay))
        self.bridge_step += 1
        return int(round(max(16, min(84, 50 + amplitude))))

    @staticmethod
    def _relative_signal(primary, reference, mode):
        dy = float(primary[1] - reference[1])
        dx = float(primary[0] - reference[0])
        distance = float(np.hypot(dx, dy))
        if mode == "distance":
            return distance
        if mode == "relative_y":
            return dy
        return (0.75 * dy) + (0.25 * distance)

    @staticmethod
    def _window_to_pos(value, window, minimum_span=18.0):
        if len(window) < 6:
            return 50
        low = min(window)
        high = max(window)
        span = high - low
        if span < 3.0:
            return 50
        effective_span = max(float(minimum_span), float(span))
        center = (low + high) * 0.5
        scaled_low = center - (effective_span * 0.5)
        norm = (float(value) - scaled_low) / max(1e-6, effective_span)
        return int(round(max(0, min(100, 100.0 - (norm * 100.0)))))

    @staticmethod
    def _suppress_short_half_cycles(actions, min_cycle_ms=320, min_delta=22):
        if len(actions) < 3:
            return actions

        filtered = [actions[0]]
        for action in actions[1:]:
            last = filtered[-1]
            if action["at"] - last["at"] < min_cycle_ms:
                if abs(action["pos"] - 50) > abs(last["pos"] - 50):
                    filtered[-1] = action
                continue
            if abs(action["pos"] - last["pos"]) < min_delta:
                if abs(action["pos"] - 50) > abs(last["pos"] - 50):
                    filtered[-1] = action
                continue
            filtered.append(action)
        return filtered

    @staticmethod
    def _finalize_actions(actions):
        actions = PeakValleyFunscriptGenerator._compress_monotonic_runs(actions)
        actions = PeakValleyFunscriptGenerator._enforce_alternating_extrema(actions)
        actions = PeakValleyFunscriptGenerator._remove_directional_micro_steps(actions)
        actions = PeakValleyFunscriptGenerator._limit_action_extremes(actions)
        return actions

    def process(self, video_path, primary_ratio, reference_ratio, signal_mode="distance"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        radius = int(max(42, min(120, min(width, height) * 0.045)))
        template_radius = int(max(24, min(54, min(width, height) * 0.020)))
        search_radius = int(max(72, min(180, min(width, height) * 0.075)))
        reacquire_search_radius = int(max(260, min(760, min(width, height) * 0.32)))

        primary = self._clip_point((primary_ratio[0] * width, primary_ratio[1] * height), width, height)
        reference = self._clip_point((reference_ratio[0] * width, reference_ratio[1] * height), width, height)
        initial_primary = primary
        initial_reference = reference
        initial_distance = float(np.hypot(primary[0] - reference[0], primary[1] - reference[1]))
        max_step = float(max(18.0, min(width, height) * 0.035))
        startup_frames = max(24, int(fps * 1.5))
        min_reacquire_frame = max(startup_frames + 15, int(fps * 2.0))
        reacquire_confirmations = 3
        reacquire_pending_drift = float(max(36.0, min(96.0, min(width, height) * 0.045)))
        reacquire_max_jump = float(max(180.0, min(420.0, min(width, height) * 0.20)))

        prev_gray = None
        primary_features = None
        reference_features = None
        primary_template = None
        reference_template = None
        primary_lock_template = None
        reference_lock_template = None
        initial_primary_template = None
        initial_reference_template = None
        initial_pair_template = None
        raw_window = deque(maxlen=max(12, int(fps * 1.6)))
        samples = []
        diagnostics = []
        last_pos = 50
        verified_streak = 0

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                at_ms = int((frame_idx / fps) * 1000)
                preview = frame.copy()
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                black_frame = self._is_black_frame(frame)

                p_count = 0
                r_count = 0
                p_retention = 0.0
                r_retention = 0.0
                p_template_score = 0.0
                r_template_score = 0.0
                p_lock_score = 0.0
                r_lock_score = 0.0
                failure_reason = ""
                tracking_quality = "init"
                pair_score = 0.0
                geometry_score = 0.0
                pair_context_score = 0.0
                reacquire_pending_count = 0
                startup_lock = frame_idx < startup_frames
                primary_step_px = 0.0
                reference_step_px = 0.0
                accept_alpha = 0.0
                accept_step = 0.0
                raw_value = self._relative_signal(primary, reference, signal_mode)

                if black_frame:
                    state = TrackingState.PATTERN_BRIDGE
                    provenance = SignalProvenance.REPLAYED_PATTERN
                    if not self.bridge_template:
                        self._seed_bridge()
                    pos = self._next_bridge_pos(last_pos, self.lost_streak)
                    confidence = 0.30
                    primary_features = None
                    reference_features = None
                    primary_template = None
                    reference_template = None
                    primary_lock_template = None
                    reference_lock_template = None
                    prev_gray = None
                    self.lost_streak += 1
                    self.pending_reacquire = None
                    verified_streak = 0
                    failure_reason = "black_frame"
                    tracking_quality = "bridge"
                elif prev_gray is None:
                    primary_features = self._seed_features(curr_gray, primary, width, height, radius)
                    reference_features = self._seed_features(curr_gray, reference, width, height, radius)
                    primary_template = self._extract_template(curr_gray, primary, template_radius)
                    reference_template = self._extract_template(curr_gray, reference, template_radius)
                    primary_lock_template = self._extract_template(curr_gray, primary, max(8, template_radius // 2), allow_flat=True)
                    reference_lock_template = self._extract_template(curr_gray, reference, max(8, template_radius // 2), allow_flat=True)
                    if initial_primary_template is None:
                        initial_primary_template = primary_template
                    if initial_reference_template is None:
                        initial_reference_template = reference_template
                    if initial_pair_template is None:
                        initial_pair_template = self._extract_pair_template(curr_gray, primary, reference)
                    p_count = 0 if primary_features is None else len(primary_features)
                    r_count = 0 if reference_features is None else len(reference_features)
                    raw_window.append(raw_value)
                    pos = self._window_to_pos(raw_value, raw_window)
                    confidence = 0.65
                    state = TrackingState.TRACKING_DEGRADED
                    provenance = SignalProvenance.TRACKED
                    self.lost_streak = 0
                    self.pending_reacquire = None
                    verified_streak = 0
                    tracking_quality = "seed"
                else:
                    next_primary, next_primary_features, p_count, p_retention, p_ok, p_reason = self._track_cloud(
                        prev_gray, curr_gray, primary_features, primary, width, height, max_step
                    )
                    next_reference, next_reference_features, r_count, r_retention, r_ok, r_reason = self._track_cloud(
                        prev_gray, curr_gray, reference_features, reference, width, height, max_step
                    )

                    lock_primary, p_lock_score, p_lock_ok, _ = self._track_locked_point(
                        curr_gray, primary_lock_template, primary, width, height, search_radius=max(24, search_radius // 2)
                    )
                    lock_reference, r_lock_score, r_lock_ok, _ = self._track_locked_point(
                        curr_gray, reference_lock_template, reference, width, height, search_radius=max(24, search_radius // 2)
                    )
                    if p_lock_ok:
                        next_primary = lock_primary
                        next_primary_features = self._seed_features(curr_gray, next_primary, width, height, radius)
                        p_count = 0 if next_primary_features is None else len(next_primary_features)
                        p_retention = max(p_retention, p_lock_score)
                        p_ok = p_count >= 4
                        p_reason = "locked_point" if p_ok else "locked_point_no_features"
                    if r_lock_ok:
                        next_reference = lock_reference
                        next_reference_features = self._seed_features(curr_gray, next_reference, width, height, radius)
                        r_count = 0 if next_reference_features is None else len(next_reference_features)
                        r_retention = max(r_retention, r_lock_score)
                        r_ok = r_count >= 4
                        r_reason = "locked_point" if r_ok else "locked_point_no_features"

                    if not p_ok:
                        template_primary, p_template_score, p_template_ok, p_template_reason = self._track_template(
                            curr_gray, primary_template, primary, width, height, search_radius
                        )
                        if p_template_ok:
                            next_primary = template_primary
                            next_primary_features = self._seed_features(curr_gray, next_primary, width, height, radius)
                            p_count = 0 if next_primary_features is None else len(next_primary_features)
                            p_retention = max(p_retention, p_template_score)
                            p_ok = p_count >= 4
                            p_reason = "template" if p_ok else "template_no_features"
                        else:
                            p_reason = p_template_reason

                    if not r_ok:
                        template_reference, r_template_score, r_template_ok, r_template_reason = self._track_template(
                            curr_gray, reference_template, reference, width, height, search_radius
                        )
                        if r_template_ok:
                            next_reference = template_reference
                            next_reference_features = self._seed_features(curr_gray, next_reference, width, height, radius)
                            r_count = 0 if next_reference_features is None else len(next_reference_features)
                            r_retention = max(r_retention, r_template_score)
                            r_ok = r_count >= 4
                            r_reason = "template" if r_ok else "template_no_features"
                        else:
                            r_reason = r_template_reason

                    pair_ok = False
                    pair_reason = "not_tracked"
                    if p_ok and r_ok:
                        pair_ok, pair_reason = self._pair_is_valid(next_primary, next_reference, initial_distance, width, height)
                        geometry_score = self._pair_geometry_score(next_primary, next_reference, initial_distance)
                        if pair_ok and (self._near_boundary(next_primary, width, height) or self._near_boundary(next_reference, width, height)):
                            pair_ok = False
                            pair_reason = "boundary_exit"

                    good = (
                        p_ok
                        and r_ok
                        and pair_ok
                        and p_count >= 6
                        and r_count >= 6
                        and p_retention >= 0.30
                        and r_retention >= 0.30
                    )
                    if not good and p_ok and r_ok and pair_ok:
                        if p_count < 6 or r_count < 6:
                            pair_reason = "low_feature_count"
                        elif p_retention < 0.30 or r_retention < 0.30:
                            pair_reason = "low_retention"

                    if good:
                        p_template_score = self._verify_template_near_point(
                            curr_gray, primary_template, next_primary, width, height, radius=36
                        )
                        r_template_score = self._verify_template_near_point(
                            curr_gray, reference_template, next_reference, width, height, radius=36
                        )
                        pair_score = self._pair_score(
                            p_retention,
                            r_retention,
                            p_template_score,
                            r_template_score,
                            geometry_score,
                        )

                        if min(max(p_template_score, p_lock_score), max(r_template_score, r_lock_score)) >= 0.42 and pair_score >= 0.58:
                            tracking_quality = "verified"
                        else:
                            good = False
                            tracking_quality = "flow_only"
                            pair_reason = "flow_only"

                    if good:
                        raw_next_primary = next_primary
                        raw_next_reference = next_reference
                        accept_alpha, accept_step = self._adaptive_acceptance(
                            startup_lock,
                            pair_score,
                            p_template_score,
                            r_template_score,
                            width,
                            height,
                        )
                        next_primary = self._smooth_candidate(primary, raw_next_primary, width, height, accept_alpha, accept_step)
                        next_reference = self._smooth_candidate(reference, raw_next_reference, width, height, accept_alpha, accept_step)
                        primary_step_px = self._point_distance(primary, next_primary)
                        reference_step_px = self._point_distance(reference, next_reference)

                        primary = next_primary
                        reference = next_reference
                        if self._point_distance(primary, raw_next_primary) > 3.0:
                            primary_features = self._seed_features(curr_gray, primary, width, height, radius)
                            p_count = 0 if primary_features is None else len(primary_features)
                        else:
                            primary_features = next_primary_features
                        if self._point_distance(reference, raw_next_reference) > 3.0:
                            reference_features = self._seed_features(curr_gray, reference, width, height, radius)
                            r_count = 0 if reference_features is None else len(reference_features)
                        else:
                            reference_features = next_reference_features

                        verified_streak += 1
                        if frame_idx % 12 == 0 or primary_features is None or len(primary_features) < 8:
                            seeded = self._seed_features(curr_gray, primary, width, height, radius)
                            if seeded is not None and len(seeded) >= max(10, p_count):
                                primary_features = seeded
                                p_count = len(seeded)
                        if frame_idx % 12 == 0 or reference_features is None or len(reference_features) < 8:
                            seeded = self._seed_features(curr_gray, reference, width, height, radius)
                            if seeded is not None and len(seeded) >= max(10, r_count):
                                reference_features = seeded
                                r_count = len(seeded)
                        allow_template_refresh = (not startup_lock) or verified_streak >= 6
                        if frame_idx % 6 == 0 or primary_lock_template is None:
                            refreshed = self._extract_template(curr_gray, primary, max(8, template_radius // 2), allow_flat=True)
                            if refreshed is not None and (p_lock_score >= 0.52 or p_template_score >= 0.58):
                                primary_lock_template = refreshed
                        if frame_idx % 6 == 0 or reference_lock_template is None:
                            refreshed = self._extract_template(curr_gray, reference, max(8, template_radius // 2), allow_flat=True)
                            if refreshed is not None and (r_lock_score >= 0.52 or r_template_score >= 0.58):
                                reference_lock_template = refreshed
                        if allow_template_refresh and (frame_idx % 14 == 0 or primary_template is None):
                            refreshed = self._extract_template(curr_gray, primary, template_radius)
                            if refreshed is not None and p_template_score >= 0.58:
                                primary_template = refreshed
                        if allow_template_refresh and (frame_idx % 14 == 0 or reference_template is None):
                            refreshed = self._extract_template(curr_gray, reference, template_radius)
                            if refreshed is not None and r_template_score >= 0.58:
                                reference_template = refreshed

                        raw_value = self._relative_signal(primary, reference, signal_mode)
                        raw_window.append(raw_value)
                        pos = self._window_to_pos(raw_value, raw_window)
                        confidence = min(1.0, 0.40 + (0.25 * min(1.0, p_count / 24.0)) + (0.25 * min(1.0, r_count / 24.0)) + (0.10 * pair_score))
                        state = TrackingState.TRACKING_CONFIDENT if confidence >= 0.78 else TrackingState.TRACKING_DEGRADED
                        provenance = SignalProvenance.TRACKED
                        self.recent_positions.append(pos)
                        last_pos = pos
                        self.lost_streak = 0
                        self.pending_reacquire = None
                        failure_reason = ""
                    else:
                        self.lost_streak += 1
                        verified_streak = 0
                        primary_features = None
                        reference_features = None
                        failure_reason = p_reason if not p_ok else (r_reason if not r_ok else pair_reason)
                        if failure_reason == "flow_only":
                            tracking_quality = "flow_only"

                        reacquired = None
                        can_reacquire = (
                            frame_idx >= min_reacquire_frame
                            and len(self.recent_positions) >= 12
                            and initial_primary_template is not None
                            and initial_reference_template is not None
                            and initial_pair_template is not None
                            and self.lost_streak >= 14
                        )
                        if can_reacquire:
                            reacquired, p_template_score, r_template_score, reacquire_reason = self._try_reacquire_initial(
                                curr_gray=curr_gray,
                                width=width,
                                height=height,
                                initial_primary=initial_primary,
                                initial_reference=initial_reference,
                                initial_primary_template=initial_primary_template,
                                initial_reference_template=initial_reference_template,
                                initial_pair_template=initial_pair_template,
                                initial_distance=initial_distance,
                                radius=radius,
                                search_radius=reacquire_search_radius,
                            )
                            if reacquired is not None:
                                pair_context_score = float(reacquired.get("pair_context_score", 0.0))
                                reacquire_jump = max(
                                    self._point_distance(primary, reacquired["primary"]),
                                    self._point_distance(reference, reacquired["reference"]),
                                )
                                if reacquire_jump > reacquire_max_jump:
                                    self.pending_reacquire = None
                                    failure_reason = "reacquire_jump"
                                else:
                                    reacquire_pending_count = self._update_pending_reacquire(
                                        reacquired,
                                        max_drift=reacquire_pending_drift,
                                    )
                                    failure_reason = "reacquire_pending"

                                if reacquire_pending_count >= reacquire_confirmations:
                                    primary = reacquired["primary"]
                                    reference = reacquired["reference"]
                                    primary_features = reacquired["primary_features"]
                                    reference_features = reacquired["reference_features"]
                                    refreshed_primary = self._extract_template(curr_gray, primary, template_radius)
                                    refreshed_reference = self._extract_template(curr_gray, reference, template_radius)
                                    primary_template = refreshed_primary if refreshed_primary is not None else initial_primary_template
                                    reference_template = refreshed_reference if refreshed_reference is not None else initial_reference_template
                                    raw_value = self._relative_signal(primary, reference, signal_mode)
                                    raw_window.append(raw_value)
                                    pos = self._window_to_pos(raw_value, raw_window)
                                    confidence = 0.74
                                    state = TrackingState.TRACKING_DEGRADED
                                    provenance = SignalProvenance.TRACKED
                                    self.lost_streak = 0
                                    verified_streak = 1
                                    failure_reason = "reacquired_initial"
                                    tracking_quality = "reacquired"
                                    pair_score = max(pair_score, min(p_template_score, r_template_score, pair_context_score))
                                    self.recent_positions.append(pos)
                                    last_pos = pos
                                    prev_gray = curr_gray
                                    good = True
                                    self.pending_reacquire = None
                            else:
                                self.pending_reacquire = None

                        if good:
                            pass
                        else:
                            state = TrackingState.PATTERN_BRIDGE if len(self.recent_positions) >= 8 else TrackingState.TRACKING_DEGRADED
                            provenance = SignalProvenance.REPLAYED_PATTERN if state == TrackingState.PATTERN_BRIDGE else SignalProvenance.TRACKED
                            if state == TrackingState.PATTERN_BRIDGE:
                                if not self.bridge_template or self.lost_streak == 1:
                                    self._seed_bridge()
                                pos = self._next_bridge_pos(last_pos, self.lost_streak)
                            else:
                                pos = last_pos
                            confidence = 0.42 if state == TrackingState.PATTERN_BRIDGE else 0.34
                            if state == TrackingState.PATTERN_BRIDGE:
                                tracking_quality = "bridge"
                            prev_gray = None

                sample = FunscriptSignalSample(
                    at_ms=at_ms,
                    pos=int(pos),
                    confidence=float(confidence),
                    tracking_state=state.value,
                    provenance=provenance.value,
                )
                samples.append(sample)

                cv2.circle(preview, primary, 8, (0, 255, 255), 2)
                cv2.putText(preview, "PRIMARY", (primary[0] + 10, primary[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.circle(preview, reference, 8, (255, 120, 0), 2)
                cv2.putText(preview, "REFERENCE", (reference[0] + 10, reference[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
                cv2.line(preview, primary, reference, (255, 255, 255), 1)
                cv2.putText(preview, f"TWO_POINT {state.value} POS:{int(pos)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                cv2.putText(preview, f"P:{p_count} R:{r_count} CONF:{confidence:.2f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
                if failure_reason:
                    cv2.putText(preview, f"FAIL:{failure_reason} LOST:{self.lost_streak}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 2)

                diagnostics.append(
                    {
                        "frame": frame_idx,
                        "at_ms": at_ms,
                        "tracking_state": state.value,
                        "signal_provenance": provenance.value,
                        "primary_x": primary[0],
                        "primary_y": primary[1],
                        "reference_x": reference[0],
                        "reference_y": reference[1],
                        "primary_features": int(p_count),
                        "reference_features": int(r_count),
                        "primary_retention": round(float(p_retention), 4),
                        "reference_retention": round(float(r_retention), 4),
                        "primary_template_score": round(float(p_template_score), 4),
                        "reference_template_score": round(float(r_template_score), 4),
                        "primary_lock_score": round(float(p_lock_score), 4),
                        "reference_lock_score": round(float(r_lock_score), 4),
                        "pair_score": round(float(pair_score), 4),
                        "geometry_score": round(float(geometry_score), 4),
                        "pair_context_score": round(float(pair_context_score), 4),
                        "tracking_quality": tracking_quality,
                        "startup_lock": int(startup_lock),
                        "verified_streak": int(verified_streak),
                        "reacquire_pending_count": int(reacquire_pending_count),
                        "initial_primary_x": initial_primary[0],
                        "initial_primary_y": initial_primary[1],
                        "initial_reference_x": initial_reference[0],
                        "initial_reference_y": initial_reference[1],
                        "initial_distance": round(float(initial_distance), 4),
                        "current_distance": round(float(np.hypot(primary[0] - reference[0], primary[1] - reference[1])), 4),
                        "primary_step_px": round(float(primary_step_px), 4),
                        "reference_step_px": round(float(reference_step_px), 4),
                        "accept_alpha": round(float(accept_alpha), 4),
                        "accept_step": round(float(accept_step), 4),
                        "lost_streak": int(self.lost_streak),
                        "failure_reason": failure_reason,
                        "raw_signal": round(float(raw_value), 4),
                        "pos": int(pos),
                        "confidence": round(float(confidence), 4),
                        "black_frame": int(black_frame),
                    }
                )

                prev_gray = curr_gray
                frame_idx += 1
                if self.callback:
                    self.callback(
                        frame_idx,
                        total_frames,
                        preview,
                        {
                            "event": "progress",
                            "stroke_pos": int(pos),
                            "tracking_state": state.value,
                        },
                    )
        finally:
            cap.release()

        actions = PeakValleyFunscriptGenerator(fps).generate(samples)
        actions = self._suppress_short_half_cycles(actions)
        actions = self._finalize_actions(actions)
        return TwoPointTrackResult(
            data={
                "version": "2.0",
                "inverted": False,
                "range": 90,
                "mode": "two_point",
                "actions": actions,
            },
            diagnostics=diagnostics,
        )
