from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SignalAlignment:
    polarity: int = 1
    amplitude_scale: float = 1.0
    phase_shift_frames: int = 0
    source_center: float = 50.0
    target_center: float = 50.0
    baseline_offset: float = 0.0
    confidence: float = 0.0

    def apply(self, raw_pos):
        centered = float(raw_pos) - float(self.source_center)
        aligned = (self.polarity * self.amplitude_scale * centered) + float(self.target_center) + float(self.baseline_offset)
        return max(0.0, min(100.0, aligned))

    def as_dict(self):
        return asdict(self)


@dataclass
class SignalStitchStep:
    position: float
    mode: str
    transition_active: bool
    transform_ready: bool
    transform_active: bool
    using_bridge: bool
    alignment_confidence: float
    polarity: int
    amplitude_scale: float
    phase_shift_frames: int
    blend_progress: float

    def as_dict(self):
        return asdict(self)


class SignalStitcher:
    def __init__(
        self,
        fps,
        warmup_seconds=0.35,
        transition_seconds=0.30,
        reference_seconds=0.80,
        max_phase_seconds=0.18,
    ):
        self.fps = max(1.0, float(fps))
        self.warmup_frames = max(8, int(round(self.fps * warmup_seconds)))
        self.min_warmup_frames = max(6, int(round(self.fps * 0.18)))
        self.reference_frames = max(self.warmup_frames * 2, int(round(self.fps * reference_seconds)))
        self.transition_frames = max(8, int(round(self.fps * transition_seconds)))
        self.max_phase_shift = max(1, int(round(self.fps * max_phase_seconds)))
        self.reset()

    def reset(self):
        self.active = False
        self.reference_history = []
        self.raw_history = []
        self.bridge_history = []
        self.alignment = None
        self.phase_extension_remaining = 0
        self.phase_delay_frames = 0
        self.delay_buffer = deque()
        self.blend_remaining = 0

    def start(self, reference_positions):
        samples = [float(pos) for pos in reference_positions if pos is not None]
        if not samples:
            samples = [50.0]

        self.active = True
        self.reference_history = samples[-self.reference_frames :]
        self.raw_history = []
        self.bridge_history = []
        self.alignment = None
        self.phase_extension_remaining = 0
        self.phase_delay_frames = 0
        self.delay_buffer = deque()
        self.blend_remaining = 0

    def is_transitioning(self):
        return self.active

    def has_transform(self):
        return self.alignment is not None

    def snapshot(self):
        if self.alignment is None:
            return {
                "mode": "inactive" if not self.active else "warmup",
                "transition_active": self.active,
                "transform_ready": False,
                "transform_active": False,
                "using_bridge": self.active,
                "alignment_confidence": 0.0,
                "polarity": 1,
                "amplitude_scale": 1.0,
                "phase_shift_frames": 0,
                "blend_progress": 0.0,
            }

        return {
            "mode": "transform" if not self.active else "blend",
            "transition_active": self.active,
            "transform_ready": True,
            "transform_active": True,
            "using_bridge": False,
            "alignment_confidence": float(self.alignment.confidence),
            "polarity": int(self.alignment.polarity),
            "amplitude_scale": float(self.alignment.amplitude_scale),
            "phase_shift_frames": int(self.alignment.phase_shift_frames),
            "blend_progress": self._blend_progress(),
        }

    def step(self, raw_pos, bridge_pos):
        raw_pos = float(raw_pos)
        bridge_pos = float(bridge_pos)

        if not self.active and self.alignment is None:
            return self._build_step(
                position=raw_pos,
                mode="inactive",
                transition_active=False,
                transform_ready=False,
                transform_active=False,
                using_bridge=False,
                blend_progress=0.0,
            )

        if self.active and self.alignment is None:
            self.raw_history.append(raw_pos)
            self.bridge_history.append(bridge_pos)

            if self._ready_for_alignment():
                self.alignment = self._estimate_alignment()
                self.phase_extension_remaining = max(0, int(self.alignment.phase_shift_frames))
                self.phase_delay_frames = max(0, -int(self.alignment.phase_shift_frames))
                self.delay_buffer = deque(maxlen=max(2, self.phase_delay_frames + 2))
                self.blend_remaining = self.transition_frames

            return self._build_step(
                position=bridge_pos,
                mode="warmup",
                transition_active=True,
                transform_ready=self.alignment is not None,
                transform_active=self.alignment is not None,
                using_bridge=True,
                blend_progress=0.0,
            )

        corrected = self.alignment.apply(raw_pos)
        using_bridge = False
        mode = "blend"

        if self.phase_delay_frames > 0:
            self.delay_buffer.append(corrected)
            if len(self.delay_buffer) <= self.phase_delay_frames:
                corrected = bridge_pos
                using_bridge = True
                mode = "phase_delay"
            else:
                corrected = float(self.delay_buffer.popleft())

        if self.phase_extension_remaining > 0:
            self.phase_extension_remaining -= 1
            corrected = bridge_pos
            using_bridge = True
            mode = "phase_hold"

        if self.active and using_bridge:
            return self._build_step(
                position=corrected,
                mode=mode,
                transition_active=True,
                transform_ready=True,
                transform_active=True,
                using_bridge=True,
                blend_progress=0.0,
            )

        if self.active and self.blend_remaining > 0:
            progress = self._blend_progress()
            blend = progress * progress * (3.0 - (2.0 * progress))
            stitched = ((1.0 - blend) * bridge_pos) + (blend * corrected)
            self.blend_remaining -= 1

            if self.blend_remaining <= 0:
                self.active = False

            return self._build_step(
                position=stitched,
                mode="blend",
                transition_active=self.active,
                transform_ready=True,
                transform_active=True,
                using_bridge=False,
                blend_progress=progress,
            )

        self.active = False
        return self._build_step(
            position=corrected,
            mode="transform",
            transition_active=False,
            transform_ready=True,
            transform_active=True,
            using_bridge=False,
            blend_progress=1.0,
        )

    def _ready_for_alignment(self):
        if len(self.raw_history) < self.min_warmup_frames:
            return False

        if len(self.raw_history) >= self.warmup_frames:
            return True

        samples = np.asarray(self.raw_history, dtype=np.float32)
        if samples.size < 4:
            return False

        amplitude = float(np.percentile(samples, 90) - np.percentile(samples, 10))
        return amplitude >= 10.0

    def _estimate_alignment(self):
        candidate = np.asarray(self.raw_history, dtype=np.float32)
        reference = np.asarray(self.reference_history[-(len(candidate) + self.max_phase_shift + 6) :], dtype=np.float32)

        best = None
        min_overlap = max(5, self.min_warmup_frames // 2)

        for polarity in (1, -1):
            for lag in range(-self.max_phase_shift, self.max_phase_shift + 1):
                ref_overlap, cand_overlap = self._lagged_overlap(reference, candidate, lag)
                if len(ref_overlap) < min_overlap:
                    continue

                ref_amp = self._robust_amplitude(ref_overlap)
                cand_amp = self._robust_amplitude(cand_overlap)
                amplitude_scale = 1.0
                if cand_amp >= 2.0:
                    amplitude_scale = float(np.clip(ref_amp / max(cand_amp, 1e-4), 0.65, 1.65))

                source_center = float(np.median(cand_overlap))
                target_center = float(np.median(ref_overlap))
                transformed = (polarity * amplitude_scale * (cand_overlap - source_center)) + target_center
                mean_offset = float(np.mean(ref_overlap - transformed))
                anchor_offset = float(ref_overlap[-1] - transformed[-1])
                baseline_offset = (0.35 * mean_offset) + (0.65 * anchor_offset)
                transformed = transformed + baseline_offset

                corr = self._normalized_correlation(ref_overlap, polarity * cand_overlap)
                mae = float(np.mean(np.abs(ref_overlap - transformed))) / 100.0
                endpoint_error = abs(float(ref_overlap[-1] - transformed[-1])) / 100.0
                lag_penalty = abs(lag) / float(max(1, self.max_phase_shift))
                score = (
                    (0.52 * max(0.0, corr))
                    + (0.25 * max(0.0, 1.0 - (mae * 4.0)))
                    + (0.13 * max(0.0, 1.0 - (endpoint_error * 5.0)))
                    + (0.10 * max(0.0, 1.0 - lag_penalty))
                )

                candidate_alignment = SignalAlignment(
                    polarity=int(polarity),
                    amplitude_scale=float(amplitude_scale),
                    phase_shift_frames=int(lag),
                    source_center=float(source_center),
                    target_center=float(target_center),
                    baseline_offset=float(np.clip(baseline_offset, -24.0, 24.0)),
                    confidence=float(np.clip(score, 0.0, 1.0)),
                )

                if best is None or candidate_alignment.confidence > best.confidence:
                    best = candidate_alignment

        if best is not None:
            return best

        last_reference = float(reference[-1]) if reference.size else 50.0
        last_candidate = float(candidate[-1]) if candidate.size else last_reference
        return SignalAlignment(
            polarity=1,
            amplitude_scale=1.0,
            phase_shift_frames=0,
            source_center=last_candidate,
            target_center=last_reference,
            baseline_offset=float(last_reference - last_candidate),
            confidence=0.0,
        )

    @staticmethod
    def _lagged_overlap(reference, candidate, lag):
        ref = np.asarray(reference, dtype=np.float32)
        cand = np.asarray(candidate, dtype=np.float32)
        if ref.size == 0 or cand.size == 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        offset = (len(ref) - len(cand)) - int(lag)
        cand_start = max(0, -offset)
        cand_end = min(len(cand), len(ref) - offset)
        if cand_end <= cand_start:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        ref_start = cand_start + offset
        ref_end = cand_end + offset
        return ref[ref_start:ref_end], cand[cand_start:cand_end]

    @staticmethod
    def _robust_amplitude(values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, 90) - np.percentile(values, 10))

    @staticmethod
    def _normalized_correlation(reference, candidate):
        ref = np.asarray(reference, dtype=np.float32)
        cand = np.asarray(candidate, dtype=np.float32)
        if ref.size == 0 or cand.size == 0:
            return 0.0

        ref = ref - float(np.mean(ref))
        cand = cand - float(np.mean(cand))
        denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        if denom <= 1e-6:
            return 0.0
        return float(np.dot(ref, cand) / denom)

    def _blend_progress(self):
        if self.transition_frames <= 0:
            return 1.0
        completed = self.transition_frames - self.blend_remaining + 1
        return max(0.0, min(1.0, completed / float(self.transition_frames)))

    def _build_step(
        self,
        *,
        position,
        mode,
        transition_active,
        transform_ready,
        transform_active,
        using_bridge,
        blend_progress,
    ):
        alignment = self.alignment or SignalAlignment()
        return SignalStitchStep(
            position=max(0.0, min(100.0, float(position))),
            mode=mode,
            transition_active=bool(transition_active),
            transform_ready=bool(transform_ready),
            transform_active=bool(transform_active),
            using_bridge=bool(using_bridge),
            alignment_confidence=float(alignment.confidence),
            polarity=int(alignment.polarity),
            amplitude_scale=float(alignment.amplitude_scale),
            phase_shift_frames=int(alignment.phase_shift_frames),
            blend_progress=float(blend_progress),
        )
