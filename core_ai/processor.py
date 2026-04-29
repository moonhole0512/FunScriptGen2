import math
import os
from collections import deque
from dataclasses import replace
from threading import Event, Lock

import cv2
import numpy as np

from tracking import (
    AppearanceMemory,
    FunscriptSignalSample,
    PeakValleyFunscriptGenerator,
    SignalProvenance,
    SignalStitcher,
    TrackingConfidenceModel,
    TrackingMetrics,
    TrackingRecoveryEngine,
    TrackingState,
    TrackingStateMachine,
)
from tracking.state_machine import TrackingStateSnapshot
from utils.logger import VersionLogger
from utils.validator import FunscriptValidator

from .dinov3_handler import DINOv3Handler
from .sam3_handler import SAM3Handler


def simplify_funscript_rdp(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm for 1D time-series Funscript data.
    Assumes points = [{"at": t, "pos": p}, ...]
    """
    if len(points) < 3:
        return points

    dmax = 0.0
    index = 0
    p1 = points[0]
    p2 = points[-1]

    t1, y1 = p1["at"], p1["pos"]
    t2, y2 = p2["at"], p2["pos"]

    for i in range(1, len(points) - 1):
        t0, y0 = points[i]["at"], points[i]["pos"]
        y_interp = y1 + (y2 - y1) * (t0 - t1) / (t2 - t1) if t2 != t1 else y1
        d = abs(y0 - y_interp)

        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        rec_results1 = simplify_funscript_rdp(points[: index + 1], epsilon)
        rec_results2 = simplify_funscript_rdp(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2

    return [points[0], points[-1]]


class VideoProcessor:
    def __init__(self, callback=None, prompt_point=None, status_callback=None):
        self.callback = callback
        self.prompt_point = prompt_point
        self.status_callback = status_callback
        self.stop_requested = False
        self.version = "3.8.0"  # Moving anchor + validation-aware generation refinements

        import csv

        self._csv = csv

        self.sam3 = None
        self.dinov3 = None

        self._emit_status(
            stage="init",
            progress=0.05,
            message="Initializing AI pipeline...",
            sam3_state="idle",
            dinov3_state="idle",
        )

        try:
            self._emit_status(
                stage="sam3_loading",
                progress=0.18,
                message="Loading SAM3...",
                sam3_state="loading",
                dinov3_state="idle",
            )
            self.sam3 = SAM3Handler()
            self._emit_status(
                stage="sam3_ready",
                progress=0.52,
                message="SAM3 ready.",
                sam3_state="ready" if self.sam3 is not None and getattr(self.sam3, "model", None) is not None else "unavailable",
                dinov3_state="idle",
            )
        except Exception as e:
            print(f"VideoProcessor: SAM3 initialization warning: {e}")
            self._emit_status(
                stage="sam3_failed",
                progress=0.52,
                message=f"SAM3 unavailable: {e}",
                sam3_state="failed",
                dinov3_state="idle",
            )

        try:
            self._emit_status(
                stage="dinov3_loading",
                progress=0.62,
                message="Loading DINOv3...",
                sam3_state="ready" if self.sam3 is not None and getattr(self.sam3, "model", None) is not None else "unavailable",
                dinov3_state="loading",
            )
            self.dinov3 = DINOv3Handler()
            self._emit_status(
                stage="dinov3_ready",
                progress=0.94,
                message="DINOv3 ready.",
                sam3_state="ready" if self.sam3 is not None and getattr(self.sam3, "model", None) is not None else "unavailable",
                dinov3_state="ready" if self.dinov3 is not None and getattr(self.dinov3, "model", None) is not None else "unavailable",
            )
        except Exception as e:
            print(f"VideoProcessor: DINOv3 initialization warning: {e}")
            self._emit_status(
                stage="dinov3_failed",
                progress=0.94,
                message=f"DINOv3 unavailable: {e}",
                sam3_state="ready" if self.sam3 is not None and getattr(self.sam3, "model", None) is not None else "unavailable",
                dinov3_state="failed",
            )

        print(
            "VideoProcessor: Handler availability -> "
            f"SAM3={'ON' if self.sam3 is not None and getattr(self.sam3, 'model', None) is not None else 'OFF'}, "
            f"DINOv3={'ON' if self.dinov3 is not None and getattr(self.dinov3, 'model', None) is not None else 'OFF'}"
        )
        self._emit_status(
            stage="ready",
            progress=1.0,
            message="AI models loaded.",
            sam3_state="ready" if self.sam3 is not None and getattr(self.sam3, "model", None) is not None else "unavailable",
            dinov3_state="ready" if self.dinov3 is not None and getattr(self.dinov3, "model", None) is not None else "unavailable",
        )

        self.validator = FunscriptValidator()
        self.logger = VersionLogger()
        self.confidence_model = TrackingConfidenceModel()
        self.state_machine = TrackingStateMachine()
        self.appearance_memory = AppearanceMemory()
        self.recovery_engine = TrackingRecoveryEngine()
        self.reannotation_event = Event()
        self.reannotation_lock = Lock()
        self.reannotation_response = None
        self.reannotation_request_id = 0
        self.signal_stitcher = None
        self.last_stitch_info = None
        self.funscript_generator = None
        self.seed_prompt_px = None
        self.initial_seed_prompt_px = None
        self.moving_anchor_px = None
        self.anchor_age_frames = 0
        self.initial_seed_reprobe_cooldown = 0

    def _emit_status(self, *, stage, progress, message, sam3_state, dinov3_state):
        if self.status_callback is None:
            return

        try:
            self.status_callback(
                {
                    "stage": stage,
                    "progress": float(progress),
                    "message": message,
                    "sam3_state": sam3_state,
                    "dinov3_state": dinov3_state,
                }
            )
        except Exception as status_error:
            print(f"VideoProcessor: status callback warning: {status_error}")

    def _reset_tracking_session(self, fps):
        self.p0 = None
        self.last_mask = None
        self.pos_integral = 0.0
        self.diag_logs = []
        self.prev_mask_area_ratio = 0.0
        self.prev_prompt_px = None
        self.current_fps = fps
        self.recent_trusted_actions = deque(maxlen=max(24, int(fps * 1.2)))
        self.last_trusted_pos = 50
        self.last_trusted_at = 0
        self.bridge_template = []
        self.bridge_step = 0
        self.manual_reannotation_cooldown = 0
        self.signal_stitcher = SignalStitcher(fps=fps)
        self.last_stitch_info = self.signal_stitcher.snapshot()
        self.funscript_generator = PeakValleyFunscriptGenerator(fps=fps)
        self.seed_prompt_px = None
        self.initial_seed_prompt_px = None
        self.moving_anchor_px = None
        self.anchor_age_frames = 0
        self.initial_seed_reprobe_cooldown = 0
        self.state_machine.reset()
        self.appearance_memory.reset()
        self.reannotation_event.clear()
        self.reannotation_response = None

    @staticmethod
    def _clamp_int(value, low=0, high=100):
        return max(low, min(high, int(round(value))))

    @staticmethod
    def _append_signal_sample(signal_samples, at_ms, pos, report, state_snapshot):
        signal_samples.append(
            FunscriptSignalSample(
                at_ms=int(at_ms),
                pos=int(pos),
                confidence=float(getattr(report, "overall_confidence", 0.0)),
                tracking_state=state_snapshot.state.value,
                provenance=state_snapshot.provenance.value,
            )
        )

    @staticmethod
    def _state_color(state):
        if state == TrackingState.TRACKING_CONFIDENT:
            return (0, 255, 0)
        if state == TrackingState.TRACKING_DEGRADED:
            return (0, 255, 255)
        if state in (TrackingState.RECOVER_LOCAL, TrackingState.RECOVER_GLOBAL):
            return (0, 165, 255)
        if state == TrackingState.PATTERN_BRIDGE:
            return (255, 255, 0)
        if state == TrackingState.USER_REANNOTATE:
            return (0, 0, 255)
        return (255, 255, 255)

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _blend_point(point_a, point_b, alpha):
        alpha = max(0.0, min(1.0, float(alpha)))
        return (
            int(round((point_a[0] * (1.0 - alpha)) + (point_b[0] * alpha))),
            int(round((point_a[1] * (1.0 - alpha)) + (point_b[1] * alpha))),
        )

    @staticmethod
    def _build_motion_map(prev_gray_small, curr_gray_small):
        if prev_gray_small is None or curr_gray_small is None:
            return None

        flow = cv2.calcOpticalFlowFarneback(prev_gray_small, curr_gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.GaussianBlur(mag, (5, 5), 0)

    def _active_anchor_point(self, curr_prompt=None):
        if self.moving_anchor_px is not None:
            return self.moving_anchor_px
        if curr_prompt is not None:
            return curr_prompt
        return self.seed_prompt_px

    def _update_moving_anchor(self, point, confidence=1.0, force=False):
        if point is None:
            self.anchor_age_frames += 1
            return

        if force or self.moving_anchor_px is None:
            self.moving_anchor_px = (int(point[0]), int(point[1]))
            self.anchor_age_frames = 0
            return

        blend_alpha = 0.75 if confidence >= 0.84 else 0.55
        self.moving_anchor_px = self._blend_point(self.moving_anchor_px, point, blend_alpha)
        self.anchor_age_frames = 0

    def _age_moving_anchor(self):
        self.anchor_age_frames += 1
        if self.initial_seed_reprobe_cooldown > 0:
            self.initial_seed_reprobe_cooldown -= 1

    def _calculate_boundary_pressure(self, point, width, height):
        if point is None:
            return 1.0

        margin = max(20.0, min(width, height) * 0.08)
        distance_to_edge = min(point[0], width - 1 - point[0], point[1], height - 1 - point[1])
        normalized_distance = max(0.0, min(1.0, distance_to_edge / margin))
        return 1.0 - normalized_distance

    def _can_pattern_bridge(self):
        if len(self.recent_trusted_actions) < 8:
            return False

        positions = [a["pos"] for a in self.recent_trusted_actions]
        amplitude = max(positions) - min(positions)
        return amplitude >= 12

    def _seed_pattern_bridge(self):
        if not self.recent_trusted_actions:
            self.bridge_template = []
            self.bridge_step = 0
            return

        template_len = min(len(self.recent_trusted_actions), max(12, int(self.current_fps * 0.35)))
        self.bridge_template = [sample["pos"] for sample in list(self.recent_trusted_actions)[-template_len:]]
        self.bridge_step = 0

    def _next_pattern_bridge_position(self, fallback):
        if not self.bridge_template:
            return self.last_trusted_pos if self.last_trusted_pos is not None else fallback

        template_len = len(self.bridge_template)
        template_index = self.bridge_step % template_len
        cycle_index = self.bridge_step // template_len
        amplitude_decay = max(0.55, 1.0 - (0.15 * cycle_index))
        base_position = self.bridge_template[template_index]
        bridged_position = 50 + ((base_position - 50) * amplitude_decay)
        self.bridge_step += 1
        return self._clamp_int(bridged_position)

    def _record_trusted_sample(self, at_ms, pos, confidence):
        if confidence < 0.70:
            return

        self.recent_trusted_actions.append({"at": at_ms, "pos": pos})
        self.last_trusted_pos = pos
        self.last_trusted_at = at_ms

    def _base_tracking_metrics(
        self,
        *,
        diag_mask_active,
        mask_area_ratio,
        mask_area_change_ratio,
        appearance_confidence,
        diag_feat_cnt,
        flow_inlier_ratio,
        centroid_jump_px,
        boundary_pressure,
        curr_prompt,
        diag_s_range,
        need_ai_refresh,
        recovery_score=0.0,
        recovery_candidate_count=0,
    ):
        return TrackingMetrics(
            mask_active=bool(diag_mask_active),
            mask_area_ratio=mask_area_ratio,
            mask_area_change_ratio=mask_area_change_ratio,
            appearance_confidence=appearance_confidence,
            feature_count=diag_feat_cnt,
            flow_inlier_ratio=flow_inlier_ratio,
            centroid_jump_px=centroid_jump_px,
            boundary_pressure=boundary_pressure,
            prompt_locked=curr_prompt is not None,
            rhythm_range=diag_s_range,
            ai_refresh_requested=need_ai_refresh,
            bridge_ready=self._can_pattern_bridge(),
            recovery_score=recovery_score,
            recovery_candidate_count=recovery_candidate_count,
        )

    def _override_snapshot(self, snapshot, mode):
        if mode == "global":
            return replace(snapshot, state=TrackingState.RECOVER_GLOBAL, provenance=SignalProvenance.RECOVERED_GLOBAL)
        if mode == "local":
            return replace(snapshot, state=TrackingState.RECOVER_LOCAL, provenance=SignalProvenance.RECOVERED_LOCAL)
        return snapshot

    def _attempt_recovery(
        self,
        mode,
        frame,
        gray_frame,
        width,
        height,
        curr_prompt,
        motion_map,
    ):
        return self.recovery_engine.attempt_recovery(
            mode=mode,
            frame=frame,
            gray_frame=gray_frame,
            width=width,
            height=height,
            anchor_point=curr_prompt,
            sam_handler=self.sam3,
            dino_handler=self.dinov3,
            appearance_memory=self.appearance_memory,
            motion_map=motion_map,
        )

    def _apply_recovery_result(self, recovery_result):
        self.last_mask = recovery_result.mask
        self.p0 = recovery_result.feature_points
        self.prev_mask_area_ratio = recovery_result.mask_area_ratio
        return recovery_result.point, recovery_result.signature

    @staticmethod
    def _build_square_mask(height, width, point, half_size=72):
        mask = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, int(point[0] - half_size))
        y0 = max(0, int(point[1] - half_size))
        x1 = min(width, int(point[0] + half_size))
        y1 = min(height, int(point[1] + half_size))
        mask[y0:y1, x0:x1] = 255
        return mask

    @staticmethod
    def _point_distance(point_a, point_b):
        return math.dist(point_a, point_b)

    @staticmethod
    def _mask_contains_point(mask, point):
        if mask is None or point is None:
            return False
        x = int(max(0, min(mask.shape[1] - 1, point[0])))
        y = int(max(0, min(mask.shape[0] - 1, point[1])))
        return bool(mask[y, x] > 0)

    def _appearance_confidence(self, signature, similarity=None):
        if signature is None:
            return 0.0

        base_confidence = float(signature.get("confidence", 0.0))
        if similarity is None and self.appearance_memory.has_reference():
            similarity = self.appearance_memory.similarity(signature.get("embedding"))

        if similarity is None:
            return max(0.0, min(1.0, base_confidence))

        return max(0.0, min(1.0, (0.25 * base_confidence) + (0.75 * max(0.0, float(similarity)))))

    def _candidate_anchor_points(self, anchor_point, width, height, search_mode="single"):
        if search_mode == "single":
            return [anchor_point]

        candidates = [anchor_point]
        if search_mode == "focused":
            focus_offsets = [
                (0, 0),
                (-24, 0),
                (24, 0),
                (0, -24),
                (0, 24),
            ]
            for dx, dy in focus_offsets:
                candidates.append((anchor_point[0] + dx, anchor_point[1] + dy))
        elif self.recovery_engine is not None:
            for dx, dy in self.recovery_engine.local_offsets:
                candidates.append((anchor_point[0] + dx, anchor_point[1] + dy))

        unique_candidates = []
        seen = set()
        for point in candidates:
            clipped = (
                int(max(0, min(width - 1, point[0]))),
                int(max(0, min(height - 1, point[1]))),
            )
            if clipped not in seen:
                seen.add(clipped)
                unique_candidates.append(clipped)
        return unique_candidates

    def _score_prompt_candidate(self, anchor_point, centroid, mask, signature, width, height, motion_map):
        mask_area_ratio = self._safe_ratio(np.count_nonzero(mask), width * height)
        mask_contains_anchor = self._mask_contains_point(mask, anchor_point)
        anchor_distance = self._point_distance(anchor_point, centroid)
        proximity_score = max(0.0, 1.0 - (anchor_distance / max(72.0, min(width, height) * 0.14)))
        area_score = max(0.0, 1.0 - (abs(mask_area_ratio - 0.03) / 0.12))
        motion_score = self.recovery_engine._score_motion(centroid, motion_map) if self.recovery_engine is not None else 0.35

        appearance_similarity = None
        if signature is not None and self.appearance_memory.has_reference():
            appearance_similarity = self.appearance_memory.similarity(signature.get("embedding"))
        appearance_score = self._appearance_confidence(signature, similarity=appearance_similarity)

        total_score = (
            (0.34 * (1.0 if mask_contains_anchor else 0.0))
            + (0.22 * proximity_score)
            + (0.22 * appearance_score)
            + (0.12 * motion_score)
            + (0.10 * area_score)
        )

        return {
            "score": float(total_score),
            "mask_area_ratio": float(mask_area_ratio),
            "mask_contains_anchor": bool(mask_contains_anchor),
            "anchor_distance": float(anchor_distance),
            "appearance_similarity": 0.0 if appearance_similarity is None else float(appearance_similarity),
            "appearance_confidence": float(appearance_score),
        }

    def _fallback_target_from_point(self, frame, gray_frame, width, height, point):
        point = (int(max(0, min(width - 1, point[0]))), int(max(0, min(height - 1, point[1]))))
        mask = self._build_square_mask(height, width, point)
        signature = self.dinov3.extract_features(frame, mask=mask, prompt_point=point) if self.dinov3 is not None else None
        if signature is None:
            signature = {"centroid": point, "confidence": 0.30, "embedding": None, "point": point}

        feature_points = cv2.goodFeaturesToTrack(
            gray_frame,
            mask=mask,
            maxCorners=50,
            qualityLevel=0.1,
            minDistance=5,
        )
        appearance_confidence = self._appearance_confidence(signature)

        return {
            "point": point,
            "signature": signature,
            "mask": mask,
            "feature_points": feature_points,
            "mask_area_ratio": self._safe_ratio(np.count_nonzero(mask), width * height),
            "score": 0.28,
            "appearance_similarity": 0.0,
            "appearance_confidence": appearance_confidence,
            "mask_contains_anchor": True,
            "fallback": True,
            "seedable": False,
        }

    def _acquire_from_point(self, frame, gray_frame, width, height, point, motion_map=None, search_mode="single", strict_anchor=False):
        anchor_point = (int(max(0, min(width - 1, point[0]))), int(max(0, min(height - 1, point[1]))))
        best = None

        if strict_anchor:
            max_mask_area_ratio = 0.12
            max_anchor_distance = max(88.0, min(width, height) * 0.12)
        else:
            max_mask_area_ratio = 0.18
            max_anchor_distance = max(132.0, min(width, height) * 0.18)

        for candidate_point in self._candidate_anchor_points(anchor_point, width, height, search_mode=search_mode):
            mask = self.sam3.get_mask(frame, candidate_point) if self.sam3 is not None else None
            if mask is None or np.count_nonzero(mask) == 0:
                continue

            signature = self.dinov3.extract_features(frame, mask=mask, prompt_point=candidate_point) if self.dinov3 is not None else None
            if signature is None:
                signature = {"centroid": candidate_point, "confidence": 0.35, "embedding": None, "point": candidate_point}

            centroid = signature.get("centroid") or candidate_point
            centroid = (int(max(0, min(width - 1, centroid[0]))), int(max(0, min(height - 1, centroid[1]))))
            score_data = self._score_prompt_candidate(anchor_point, centroid, mask, signature, width, height, motion_map)

            feature_points = cv2.goodFeaturesToTrack(
                gray_frame,
                mask=mask,
                maxCorners=50,
                qualityLevel=0.1,
                minDistance=5,
            )

            candidate_result = {
                "point": centroid,
                "signature": signature,
                "mask": mask,
                "feature_points": feature_points,
                "mask_area_ratio": score_data["mask_area_ratio"],
                "score": score_data["score"],
                "appearance_similarity": score_data["appearance_similarity"],
                "appearance_confidence": score_data["appearance_confidence"],
                "mask_contains_anchor": score_data["mask_contains_anchor"],
                "fallback": False,
                "seedable": score_data["mask_contains_anchor"] and score_data["score"] >= 0.50,
            }

            if candidate_result["mask_area_ratio"] > max_mask_area_ratio:
                continue

            if strict_anchor and not candidate_result["mask_contains_anchor"]:
                continue

            if (
                strict_anchor
                and score_data["anchor_distance"] > max_anchor_distance
                and candidate_result["appearance_similarity"] < 0.72
            ):
                continue

            if best is None or candidate_result["score"] > best["score"]:
                best = candidate_result

        threshold = 0.48 if strict_anchor else 0.42
        if not self.appearance_memory.has_reference():
            threshold = max(threshold, 0.52 if strict_anchor else 0.50)

        if best is None or best["score"] < threshold:
            return self._fallback_target_from_point(frame, gray_frame, width, height, anchor_point)

        return best

    @staticmethod
    def _should_probe_seed_target(primary_target):
        if primary_target is None:
            return True

        if primary_target.get("fallback"):
            return True

        if float(primary_target.get("appearance_confidence", 0.0)) < 0.58:
            return True

        if float(primary_target.get("score", 0.0)) < 0.62:
            return True

        if float(primary_target.get("mask_area_ratio", 0.0)) > 0.08:
            return True

        return False

    def _should_probe_initial_seed(self, primary_target):
        if self.initial_seed_prompt_px is None:
            return False

        if self.initial_seed_reprobe_cooldown > 0:
            return False

        if self.anchor_age_frames < max(12, int(self.current_fps * 0.8)):
            return False

        if primary_target is not None and primary_target.get("fallback") is False:
            if float(primary_target.get("appearance_confidence", 0.0)) >= 0.52:
                return False
            if float(primary_target.get("score", 0.0)) >= 0.56:
                return False

        return True

    def _default_stitch_info(self):
        if self.signal_stitcher is None:
            return {
                "mode": "inactive",
                "transition_active": False,
                "transform_ready": False,
                "transform_active": False,
                "using_bridge": False,
                "alignment_confidence": 0.0,
                "polarity": 1,
                "amplitude_scale": 1.0,
                "phase_shift_frames": 0,
                "blend_progress": 0.0,
            }
        return self.signal_stitcher.snapshot()

    def _start_user_stitch(self):
        if not self.bridge_template:
            self._seed_pattern_bridge()

        reference_positions = [sample["pos"] for sample in self.recent_trusted_actions]
        if not reference_positions:
            reference_positions = [self.last_trusted_pos if self.last_trusted_pos is not None else 50]

        if self.signal_stitcher is None:
            self.signal_stitcher = SignalStitcher(fps=self.current_fps)

        self.signal_stitcher.start(reference_positions)
        self.last_stitch_info = self.signal_stitcher.snapshot()

    def _apply_user_stitch(self, raw_pos):
        if self.signal_stitcher is None:
            return raw_pos, False, self._default_stitch_info()

        if not self.signal_stitcher.is_transitioning() and not self.signal_stitcher.has_transform():
            info = self.signal_stitcher.snapshot()
            self.last_stitch_info = info
            return raw_pos, False, info

        bridge_pos = raw_pos
        if self.signal_stitcher.is_transitioning():
            bridge_pos = self._next_pattern_bridge_position(self.last_trusted_pos if self.last_trusted_pos is not None else raw_pos)

        stitch_step = self.signal_stitcher.step(raw_pos, bridge_pos)
        stitch_info = stitch_step.as_dict()
        self.last_stitch_info = stitch_info

        stitch_applied = stitch_info["transition_active"] or stitch_info["transform_active"]
        return self._clamp_int(stitch_step.position), stitch_applied, stitch_info

    def submit_reannotation(self, point=None, action="resume"):
        with self.reannotation_lock:
            self.reannotation_response = {"point": point, "action": action}
            self.reannotation_event.set()

    def _request_manual_reannotation(self, frame_idx, total_frames, frame_bgr, at_ms, width, height, curr_prompt, motion_map):
        if self.callback is None:
            return {"point": None, "action": "bridge"}

        candidate_points = self.recovery_engine.suggest_points(
            width=width,
            height=height,
            anchor_point=curr_prompt,
            motion_map=motion_map,
            limit=6,
        )

        with self.reannotation_lock:
            self.reannotation_request_id += 1
            request_id = self.reannotation_request_id
            self.reannotation_response = None
            self.reannotation_event.clear()

        self.callback(
            frame_idx,
            total_frames,
            frame_bgr,
            {
                "event": "manual_reannotation",
                "request_id": request_id,
                "at_ms": at_ms,
                "current_point": curr_prompt,
                "candidate_points": candidate_points,
            },
        )

        did_receive = self.reannotation_event.wait(timeout=300.0)
        if not did_receive:
            return {"point": None, "action": "bridge"}

        with self.reannotation_lock:
            response = self.reannotation_response or {"point": None, "action": "bridge"}
            self.reannotation_response = None
        return response

    def _append_diag_log(
        self,
        frame_idx,
        at_ms,
        metrics,
        report,
        state_snapshot,
        velocity_y,
        pos_to_append,
        diag_s_range,
        stitch_info=None,
    ):
        stitch_info = stitch_info or self._default_stitch_info()
        self.diag_logs.append(
            {
                "frame": frame_idx,
                "at_ms": at_ms,
                "tracking_state": state_snapshot.state.value,
                "signal_provenance": state_snapshot.provenance.value,
                "user_intervention_required": int(state_snapshot.user_intervention_required),
                "overall_confidence": round(float(report.overall_confidence), 4),
                "mask_active": int(metrics.mask_active),
                "mask_area_ratio": round(float(metrics.mask_area_ratio), 6),
                "mask_area_change_ratio": round(float(metrics.mask_area_change_ratio), 4),
                "appearance_confidence": round(float(metrics.appearance_confidence), 4),
                "feat_cnt": int(metrics.feature_count),
                "flow_inlier_ratio": round(float(metrics.flow_inlier_ratio), 4),
                "centroid_jump_px": round(float(metrics.centroid_jump_px), 4),
                "boundary_pressure": round(float(metrics.boundary_pressure), 4),
                "bridge_ready": int(metrics.bridge_ready),
                "recovery_score": round(float(metrics.recovery_score), 4),
                "recovery_candidate_count": int(metrics.recovery_candidate_count),
                "vel_y": round(float(velocity_y), 4),
                "int_y": round(float(self.pos_integral), 4),
                "s_range": round(float(diag_s_range), 4),
                "pos": int(pos_to_append),
                "stitch_mode": stitch_info["mode"],
                "stitch_transition_active": int(stitch_info["transition_active"]),
                "stitch_transform_ready": int(stitch_info["transform_ready"]),
                "stitch_transform_active": int(stitch_info["transform_active"]),
                "stitch_using_bridge": int(stitch_info["using_bridge"]),
                "stitch_alignment_confidence": round(float(stitch_info["alignment_confidence"]), 4),
                "stitch_polarity": int(stitch_info["polarity"]),
                "stitch_amplitude_scale": round(float(stitch_info["amplitude_scale"]), 4),
                "stitch_phase_frames": int(stitch_info["phase_shift_frames"]),
                "stitch_blend_progress": round(float(stitch_info["blend_progress"]), 4),
            }
        )

    def _summarize_tracking(self):
        if not self.diag_logs:
            return {}

        total_rows = len(self.diag_logs)
        trusted_rows = sum(1 for row in self.diag_logs if row["tracking_state"] == TrackingState.TRACKING_CONFIDENT.value)
        bridge_rows = sum(1 for row in self.diag_logs if row["signal_provenance"] == SignalProvenance.REPLAYED_PATTERN.value)
        local_rows = sum(1 for row in self.diag_logs if row["signal_provenance"] == SignalProvenance.RECOVERED_LOCAL.value)
        global_rows = sum(1 for row in self.diag_logs if row["signal_provenance"] == SignalProvenance.RECOVERED_GLOBAL.value)
        reannotate_rows = sum(1 for row in self.diag_logs if row["user_intervention_required"] == 1)
        stitch_transition_rows = sum(1 for row in self.diag_logs if row["stitch_transition_active"] == 1)
        stitch_transform_rows = sum(1 for row in self.diag_logs if row["stitch_transform_active"] == 1)
        stitch_ready_rows = [row["stitch_alignment_confidence"] for row in self.diag_logs if row["stitch_transform_ready"] == 1]
        confidence_mean = sum(row["overall_confidence"] for row in self.diag_logs) / total_rows

        return {
            "trusted_tracking_ratio": round(trusted_rows / total_rows, 4),
            "pattern_bridge_ratio": round(bridge_rows / total_rows, 4),
            "local_recovery_ratio": round(local_rows / total_rows, 4),
            "global_recovery_ratio": round(global_rows / total_rows, 4),
            "manual_review_ratio": round(reannotate_rows / total_rows, 4),
            "mean_confidence": round(confidence_mean, 4),
            "stitch_transition_ratio": round(stitch_transition_rows / total_rows, 4),
            "stitch_transform_ratio": round(stitch_transform_rows / total_rows, 4),
            "mean_stitch_confidence": round(sum(stitch_ready_rows) / len(stitch_ready_rows), 4) if stitch_ready_rows else 0.0,
        }

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps == 0:
            fps = 30

        self._reset_tracking_session(fps)

        curr_prompt = None
        if self.prompt_point:
            curr_prompt = (int(self.prompt_point[0] * width), int(self.prompt_point[1] * height))

        signal_samples = []
        frame_idx = 0
        prev_gray = None
        prev_gray_full = None
        target_acquired = curr_prompt is not None
        if target_acquired:
            self.prev_prompt_px = curr_prompt
            self.seed_prompt_px = curr_prompt
            self.initial_seed_prompt_px = curr_prompt
            self.moving_anchor_px = curr_prompt

        motion_heatmap = np.zeros((height // 4, width // 4), dtype=np.float32)
        init_frames_limit = 15
        signal_window = deque(maxlen=int(fps * 2))

        try:
            while cap.isOpened() and not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                at_ms = int((frame_idx / fps) * 1000)
                preview_frame = frame.copy()
                small_frame = cv2.resize(frame, (width // 4, height // 4))
                curr_gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                motion_map = self._build_motion_map(prev_gray, curr_gray_small)

                if not target_acquired:
                    cv2.putText(
                        preview_frame,
                        "ANALYZING MOTION ENERGY...",
                        (width // 2 - 200, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )

                    if motion_map is not None:
                        motion_heatmap += motion_map

                    if frame_idx >= init_frames_limit:
                        _, _, _, max_loc = cv2.minMaxLoc(cv2.blur(motion_heatmap, (5, 5)))
                        target_acquired = True
                        curr_prompt = (max_loc[0] * 4, max_loc[1] * 4)
                        self.prev_prompt_px = curr_prompt
                        self.moving_anchor_px = curr_prompt
                        print(f"Motion Energy Target Acquired at: {curr_prompt}")

                    neutral_metrics = TrackingMetrics()
                    neutral_report = self.confidence_model.evaluate(neutral_metrics)
                    init_state = self.state_machine.advance(neutral_report, target_acquired=False, can_pattern_bridge=False)
                    self._append_diag_log(
                        frame_idx=frame_idx,
                        at_ms=at_ms,
                        metrics=neutral_metrics,
                        report=neutral_report,
                        state_snapshot=init_state,
                        velocity_y=0.0,
                        pos_to_append=50,
                        diag_s_range=0.0,
                        stitch_info=self._default_stitch_info(),
                    )

                    self._append_signal_sample(signal_samples, at_ms, 50, neutral_report, init_state)
                    prev_gray = curr_gray_small
                    frame_idx += 1
                    if self.callback:
                        self.callback(frame_idx, total_frames, preview_frame)
                    continue

                diag_mask_active = 0
                diag_feat_cnt = 0
                diag_s_range = 0.0
                flow_inlier_ratio = 0.0
                appearance_confidence = 0.0
                mask_area_ratio = 0.0
                mask_area_change_ratio = 1.0
                tracked_feat_count = 0
                velocity_y = 0.0
                current_signature = None
                stitch_info = self._default_stitch_info()

                curr_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                need_ai_refresh = (frame_idx % 5 == 0) or self.p0 is None or len(self.p0) < 5
                anchor_search_point = self._active_anchor_point(curr_prompt)

                if self.p0 is not None and prev_gray_full is not None:
                    prev_feature_count = len(self.p0)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray_full, curr_gray_full, self.p0, None)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]
                        tracked_feat_count = len(good_new)
                        diag_feat_cnt = tracked_feat_count
                        flow_inlier_ratio = self._safe_ratio(tracked_feat_count, prev_feature_count)
                        if tracked_feat_count > 0:
                            dy_points = good_new[:, 1] - good_old[:, 1]
                            velocity_y = float(np.median(dy_points))
                            self.p0 = good_new.reshape(-1, 1, 2)
                        else:
                            self.p0 = None
                            need_ai_refresh = True
                    else:
                        self.p0 = None
                        need_ai_refresh = True

                if need_ai_refresh and anchor_search_point is not None:
                    acquired_target = self._acquire_from_point(
                        frame=frame,
                        gray_frame=curr_gray_full,
                        width=width,
                        height=height,
                        point=anchor_search_point,
                        motion_map=motion_map,
                        search_mode="single",
                        strict_anchor=self.moving_anchor_px is not None,
                    )

                    if (
                        self.moving_anchor_px is not None
                        and self._should_probe_seed_target(acquired_target)
                    ):
                        seed_target = self._acquire_from_point(
                            frame=frame,
                            gray_frame=curr_gray_full,
                            width=width,
                            height=height,
                            point=self.moving_anchor_px,
                            motion_map=motion_map,
                            search_mode="focused",
                            strict_anchor=True,
                        )
                        if seed_target is not None:
                            if (
                                seed_target.get("fallback") is False
                                and (
                                    float(seed_target.get("score", 0.0)) >= float(acquired_target.get("score", 0.0)) + 0.08
                                    or float(seed_target.get("appearance_confidence", 0.0)) >= float(acquired_target.get("appearance_confidence", 0.0)) + 0.10
                                )
                            ):
                                acquired_target = seed_target

                    if self._should_probe_initial_seed(acquired_target):
                        initial_seed_target = self._acquire_from_point(
                            frame=frame,
                            gray_frame=curr_gray_full,
                            width=width,
                            height=height,
                            point=self.initial_seed_prompt_px,
                            motion_map=motion_map,
                            search_mode="focused",
                            strict_anchor=False,
                        )
                        if initial_seed_target is not None and initial_seed_target.get("fallback") is False:
                            if float(initial_seed_target.get("appearance_confidence", 0.0)) >= float(acquired_target.get("appearance_confidence", 0.0)) + 0.14:
                                acquired_target = initial_seed_target
                        self.initial_seed_reprobe_cooldown = max(24, int(self.current_fps * 1.2))

                    if acquired_target is not None:
                        previous_mask_area_ratio = self.prev_mask_area_ratio
                        self.last_mask = acquired_target["mask"]
                        self.p0 = acquired_target["feature_points"]
                        current_signature = acquired_target["signature"]
                        curr_prompt = acquired_target["point"]

                        diag_mask_active = 1
                        mask_area_ratio = acquired_target["mask_area_ratio"]
                        if previous_mask_area_ratio > 0.0:
                            mask_area_change_ratio = abs(mask_area_ratio - previous_mask_area_ratio) / previous_mask_area_ratio
                        else:
                            mask_area_change_ratio = 0.0

                        appearance_confidence = max(
                            0.0,
                            min(1.0, (0.65 * acquired_target["appearance_confidence"]) + (0.35 * acquired_target["score"])),
                        )
                        diag_feat_cnt = len(self.p0) if self.p0 is not None else diag_feat_cnt
                        self.prev_mask_area_ratio = mask_area_ratio

                        if not self.appearance_memory.has_reference() and acquired_target["seedable"] and current_signature is not None:
                            self.appearance_memory.observe(current_signature, confidence=1.0)
                elif self.last_mask is not None:
                    diag_mask_active = 1
                    mask_pixels = int(np.count_nonzero(self.last_mask))
                    mask_area_ratio = self._safe_ratio(mask_pixels, width * height)
                    mask_area_change_ratio = 0.0

                if self.last_mask is not None:
                    mask_indices = self.last_mask > 0
                    if np.any(mask_indices):
                        preview_frame[mask_indices] = [255, 255, 0]

                centroid_jump_px = 0.0
                if curr_prompt is not None and self.prev_prompt_px is not None:
                    centroid_jump_px = math.dist(curr_prompt, self.prev_prompt_px)

                boundary_pressure = self._calculate_boundary_pressure(curr_prompt, width, height)

                self.pos_integral += velocity_y
                self.pos_integral *= 0.995
                signal_window.append(self.pos_integral)

                pos_to_append = 50
                if len(signal_window) > 5:
                    s_min = min(signal_window)
                    s_max = max(signal_window)
                    diag_s_range = s_max - s_min

                    if diag_s_range > 5.0:
                        raw_norm = (self.pos_integral - s_min) / diag_s_range
                        pos_to_append = max(0, min(100, 100 - int(raw_norm * 100)))

                metrics = self._base_tracking_metrics(
                    diag_mask_active=diag_mask_active,
                    mask_area_ratio=mask_area_ratio,
                    mask_area_change_ratio=mask_area_change_ratio,
                    appearance_confidence=appearance_confidence,
                    diag_feat_cnt=diag_feat_cnt,
                    flow_inlier_ratio=flow_inlier_ratio,
                    centroid_jump_px=centroid_jump_px,
                    boundary_pressure=boundary_pressure,
                    curr_prompt=curr_prompt,
                    diag_s_range=diag_s_range,
                    need_ai_refresh=need_ai_refresh,
                )
                report = self.confidence_model.evaluate(metrics)
                state_snapshot = self.state_machine.advance(
                    report,
                    target_acquired=True,
                    can_pattern_bridge=metrics.bridge_ready,
                )

                if state_snapshot.user_intervention_required and self.manual_reannotation_cooldown > 0:
                    self.manual_reannotation_cooldown -= 1
                    state_snapshot = replace(
                        state_snapshot,
                        state=TrackingState.PATTERN_BRIDGE,
                        provenance=SignalProvenance.REPLAYED_PATTERN,
                        user_intervention_required=False,
                    )

                recovery_result = None
                if state_snapshot.state == TrackingState.RECOVER_LOCAL:
                    recovery_result = self._attempt_recovery(
                        "local",
                        frame,
                        curr_gray_full,
                        width,
                        height,
                        curr_prompt,
                        motion_map,
                    )
                    if recovery_result is None:
                        recovery_result = self._attempt_recovery(
                            "global",
                            frame,
                            curr_gray_full,
                            width,
                            height,
                            curr_prompt,
                            motion_map,
                        )
                        if recovery_result is not None:
                            state_snapshot = self._override_snapshot(state_snapshot, "global")
                elif state_snapshot.state == TrackingState.RECOVER_GLOBAL:
                    recovery_result = self._attempt_recovery(
                        "global",
                        frame,
                        curr_gray_full,
                        width,
                        height,
                        curr_prompt,
                        motion_map,
                    )

                if recovery_result is not None:
                    curr_prompt, current_signature = self._apply_recovery_result(recovery_result)
                    diag_mask_active = 1
                    mask_area_ratio = recovery_result.mask_area_ratio
                    mask_area_change_ratio = 0.0
                    appearance_confidence = self._appearance_confidence(
                        current_signature,
                        similarity=float(recovery_result.appearance_similarity),
                    )
                    diag_feat_cnt = len(self.p0) if self.p0 is not None else 0
                    flow_inlier_ratio = max(flow_inlier_ratio, min(1.0, diag_feat_cnt / 24.0))
                    centroid_jump_px = 0.0 if self.prev_prompt_px is None else math.dist(curr_prompt, self.prev_prompt_px)
                    boundary_pressure = self._calculate_boundary_pressure(curr_prompt, width, height)

                    metrics = self._base_tracking_metrics(
                        diag_mask_active=diag_mask_active,
                        mask_area_ratio=mask_area_ratio,
                        mask_area_change_ratio=mask_area_change_ratio,
                        appearance_confidence=appearance_confidence,
                        diag_feat_cnt=diag_feat_cnt,
                        flow_inlier_ratio=flow_inlier_ratio,
                        centroid_jump_px=centroid_jump_px,
                        boundary_pressure=boundary_pressure,
                        curr_prompt=curr_prompt,
                        diag_s_range=diag_s_range,
                        need_ai_refresh=False,
                        recovery_score=recovery_result.score,
                        recovery_candidate_count=recovery_result.candidate_count,
                    )
                    report = self.confidence_model.evaluate(metrics)

                if recovery_result is None and state_snapshot.user_intervention_required:
                    manual_response = self._request_manual_reannotation(
                        frame_idx=frame_idx,
                        total_frames=total_frames,
                        frame_bgr=frame.copy(),
                        at_ms=at_ms,
                        width=width,
                        height=height,
                        curr_prompt=self._active_anchor_point(curr_prompt),
                        motion_map=motion_map,
                    )

                    action = manual_response.get("action", "bridge")
                    if action == "abort":
                        self.stop_requested = True
                        break

                    if action == "resume" and manual_response.get("point") is not None:
                        manual_target = self._acquire_from_point(
                            frame=frame,
                            gray_frame=curr_gray_full,
                            width=width,
                            height=height,
                            point=manual_response["point"],
                            motion_map=motion_map,
                            search_mode="focused",
                            strict_anchor=True,
                        )
                        curr_prompt = manual_target["point"]
                        self.seed_prompt_px = manual_response["point"]
                        self.initial_seed_prompt_px = manual_response["point"]
                        self.moving_anchor_px = manual_target["point"]
                        self.anchor_age_frames = 0
                        current_signature = manual_target["signature"]
                        self.last_mask = manual_target["mask"]
                        self.p0 = manual_target["feature_points"]
                        self.prev_mask_area_ratio = manual_target["mask_area_ratio"]

                        diag_mask_active = 1
                        mask_area_ratio = manual_target["mask_area_ratio"]
                        mask_area_change_ratio = 0.0
                        appearance_confidence = max(
                            0.0,
                            min(1.0, (0.65 * manual_target["appearance_confidence"]) + (0.35 * manual_target["score"])),
                        )
                        diag_feat_cnt = len(self.p0) if self.p0 is not None else 0
                        flow_inlier_ratio = max(flow_inlier_ratio, min(1.0, diag_feat_cnt / 24.0))
                        centroid_jump_px = 0.0 if self.prev_prompt_px is None else math.dist(curr_prompt, self.prev_prompt_px)
                        boundary_pressure = self._calculate_boundary_pressure(curr_prompt, width, height)

                        if current_signature is not None:
                            self.appearance_memory.observe(current_signature, confidence=1.0)

                        metrics = self._base_tracking_metrics(
                            diag_mask_active=diag_mask_active,
                            mask_area_ratio=mask_area_ratio,
                            mask_area_change_ratio=mask_area_change_ratio,
                            appearance_confidence=appearance_confidence,
                            diag_feat_cnt=diag_feat_cnt,
                            flow_inlier_ratio=flow_inlier_ratio,
                            centroid_jump_px=centroid_jump_px,
                            boundary_pressure=boundary_pressure,
                            curr_prompt=curr_prompt,
                            diag_s_range=diag_s_range,
                            need_ai_refresh=False,
                            recovery_score=1.0,
                            recovery_candidate_count=1,
                        )
                        report = self.confidence_model.evaluate(metrics)
                        state_snapshot = replace(
                            state_snapshot,
                            state=TrackingState.TRACKING_DEGRADED,
                            provenance=SignalProvenance.USER_REANNOTATED,
                            user_intervention_required=False,
                        )
                        self._start_user_stitch()
                    else:
                        self.manual_reannotation_cooldown = max(12, int(self.current_fps * 0.6))
                        state_snapshot = replace(
                            state_snapshot,
                            state=TrackingState.PATTERN_BRIDGE,
                            provenance=SignalProvenance.REPLAYED_PATTERN,
                            user_intervention_required=False,
                        )

                if state_snapshot.state_changed and state_snapshot.state == TrackingState.PATTERN_BRIDGE:
                    self._seed_pattern_bridge()

                if state_snapshot.state == TrackingState.PATTERN_BRIDGE:
                    if not self.bridge_template:
                        self._seed_pattern_bridge()
                    pos_to_append = self._next_pattern_bridge_position(pos_to_append)
                elif state_snapshot.state == TrackingState.USER_REANNOTATE:
                    pos_to_append = self.last_trusted_pos

                pos_to_append, stitch_active, stitch_info = self._apply_user_stitch(pos_to_append)
                if stitch_active:
                    if state_snapshot.provenance == SignalProvenance.TRACKED or stitch_info["transition_active"]:
                        state_snapshot = replace(
                            state_snapshot,
                            provenance=SignalProvenance.USER_REANNOTATED,
                            user_intervention_required=False,
                        )

                self._record_trusted_sample(at_ms, pos_to_append, report.overall_confidence)
                if curr_prompt is not None:
                    if report.overall_confidence >= 0.72 or recovery_result is not None:
                        self._update_moving_anchor(curr_prompt, confidence=report.overall_confidence)
                    else:
                        self._age_moving_anchor()
                else:
                    self._age_moving_anchor()

                if current_signature is not None:
                    self.appearance_memory.observe(current_signature, confidence=report.overall_confidence)

                self._append_diag_log(
                    frame_idx=frame_idx,
                    at_ms=at_ms,
                    metrics=metrics,
                    report=report,
                    state_snapshot=state_snapshot,
                    velocity_y=velocity_y,
                    pos_to_append=pos_to_append,
                    diag_s_range=diag_s_range,
                    stitch_info=stitch_info,
                )

                cX, cY = curr_prompt if curr_prompt is not None else (width // 2, height // 2)
                color = self._state_color(state_snapshot.state)
                if self.initial_seed_prompt_px is not None:
                    seed_x = int(max(0, min(width - 1, self.initial_seed_prompt_px[0])))
                    seed_y = int(max(0, min(height - 1, self.initial_seed_prompt_px[1])))
                    cv2.circle(preview_frame, (seed_x, seed_y), 5, (140, 140, 140), 1)
                    cv2.putText(preview_frame, "INIT", (seed_x + 8, seed_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
                if self.moving_anchor_px is not None:
                    anchor_x = int(max(0, min(width - 1, self.moving_anchor_px[0])))
                    anchor_y = int(max(0, min(height - 1, self.moving_anchor_px[1])))
                    cv2.circle(preview_frame, (anchor_x, anchor_y), 7, (255, 80, 220), 2)
                    cv2.putText(preview_frame, "ANCHOR", (anchor_x + 10, anchor_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 220), 1)
                crosshair_len = 15
                cv2.line(preview_frame, (int(cX) - crosshair_len, int(cY)), (int(cX) + crosshair_len, int(cY)), color, 2)
                cv2.line(preview_frame, (int(cX), int(cY) - crosshair_len), (int(cX), int(cY) + crosshair_len), color, 2)

                gauge_x = 22
                gauge_y0 = 34
                gauge_y1 = 190
                gauge_w = 16
                gauge_fill_y = int(gauge_y1 - ((pos_to_append / 100.0) * (gauge_y1 - gauge_y0)))
                cv2.rectangle(preview_frame, (gauge_x, gauge_y0), (gauge_x + gauge_w, gauge_y1), (32, 42, 54), -1)
                cv2.rectangle(preview_frame, (gauge_x, gauge_fill_y), (gauge_x + gauge_w, gauge_y1), color, -1)
                cv2.rectangle(preview_frame, (gauge_x, gauge_y0), (gauge_x + gauge_w, gauge_y1), (230, 230, 230), 1)
                cv2.putText(preview_frame, "TOP", (gauge_x + 24, gauge_y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
                cv2.putText(preview_frame, "BOTTOM", (gauge_x + 24, gauge_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
                cv2.putText(preview_frame, state_snapshot.state.value, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

                diag_text = (
                    f"CONF:{report.overall_confidence:.2f} "
                    f"FEAT:{metrics.feature_count:02d} "
                    f"FLOW:{metrics.flow_inlier_ratio:.2f} "
                    f"RANGE:{diag_s_range:.1f}"
                )
                cv2.putText(preview_frame, diag_text, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

                provenance_text = f"PROV:{state_snapshot.provenance.value}"
                cv2.putText(preview_frame, provenance_text, (10, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                if recovery_result is not None:
                    recovery_text = f"RECOVERY:{recovery_result.mode.upper()} SCORE:{recovery_result.score:.2f}"
                    cv2.putText(preview_frame, recovery_text, (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                elif state_snapshot.user_intervention_required:
                    cv2.putText(
                        preview_frame,
                        "MANUAL TARGET REVIEW NEEDED",
                        (10, 116),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                if stitch_info["transform_active"] or stitch_info["transition_active"]:
                    stitch_text = (
                        f"STITCH:{stitch_info['mode'].upper()} "
                        f"P:{stitch_info['polarity']:+d} "
                        f"S:{stitch_info['amplitude_scale']:.2f} "
                        f"PH:{stitch_info['phase_shift_frames']:+d} "
                        f"C:{stitch_info['alignment_confidence']:.2f}"
                    )
                    cv2.putText(preview_frame, stitch_text, (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)

                self._append_signal_sample(signal_samples, at_ms, pos_to_append, report, state_snapshot)
                prev_gray_full = curr_gray_full
                prev_gray = curr_gray_small
                self.prev_prompt_px = curr_prompt

                frame_idx += 1
                if self.callback:
                    self.callback(
                        frame_idx,
                        total_frames,
                        preview_frame,
                        {
                            "event": "progress",
                            "stroke_pos": int(pos_to_append),
                            "tracking_state": state_snapshot.state.value,
                        },
                    )

        except Exception as e:
            print(f"CRITICAL ERROR during process loop at frame {frame_idx}: {e}")
            import traceback

            traceback.print_exc()

        cap.release()

        if self.diag_logs:
            try:
                diag_path = os.path.join("Video", f"Diagnostic_{os.path.basename(video_path)}.csv")
                with open(diag_path, "w", newline="") as f:
                    writer = self._csv.DictWriter(f, fieldnames=self.diag_logs[0].keys())
                    writer.writeheader()
                    writer.writerows(self.diag_logs)
                print(f"Diagnostic report saved: {diag_path}")
            except Exception as e:
                print(f"Failed to save diagnostic report: {e}")

        if self.stop_requested:
            return None

        funscript_actions = (
            self.funscript_generator.generate(signal_samples)
            if self.funscript_generator is not None
            else [{"at": int(sample.at_ms), "pos": int(sample.pos)} for sample in signal_samples]
        )
        print(
            "Post-Processing: Peak/Valley generator reduced samples "
            f"from {len(signal_samples)} to {len(funscript_actions)} actions."
        )

        result = {"actions": funscript_actions}

        scores = {}
        video_name = os.path.basename(video_path)
        gt_filename = os.path.splitext(video_name)[0] + ".funscript"
        gt_path = os.path.join("Video", "HumanMade", gt_filename)

        if os.path.exists(gt_path):
            scores = self.validator.calculate_score(result, gt_path)

        tracking_summary = self._summarize_tracking()
        self.logger.log_run(
            version=self.version,
            tech_stack="Motion Heatmap + SAM3 + DINOv3 + Moving Target Anchor + Peak/Valley Signal Generator + Signal Stitcher",
            changes=(
                "Added moving user-anchor tracking, reduced fixed-seed reprobe dependence, "
                "start-phase snapping, and lower-envelope signal correction."
            ),
            validation_scores=scores,
            tracking_summary=tracking_summary,
        )

        return result

    def stop(self):
        self.stop_requested = True
