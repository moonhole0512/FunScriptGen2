from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .models import SignalProvenance, TrackingState


@dataclass
class FunscriptSignalSample:
    at_ms: int
    pos: int
    confidence: float = 0.0
    tracking_state: str = TrackingState.INIT.value
    provenance: str = SignalProvenance.TRACKED.value


class PeakValleyFunscriptGenerator:
    def __init__(self, fps):
        self.fps = max(1.0, float(fps))

    @staticmethod
    def _clamp_int(value, low=0, high=100):
        return max(low, min(high, int(round(value))))

    @staticmethod
    def _odd_window(length, preferred):
        preferred = max(5, int(preferred))
        if preferred % 2 == 0:
            preferred += 1

        if length < 5:
            return 0

        if preferred >= length:
            preferred = length - 1 if length % 2 == 0 else length

        if preferred < 5:
            return 0

        if preferred % 2 == 0:
            preferred -= 1

        return preferred if preferred >= 5 else 0

    def _smooth_signal(self, signal):
        if len(signal) < 5:
            return signal

        preferred = max(9, int(round(self.fps * 0.18)))
        window = self._odd_window(len(signal), preferred)
        if window == 0:
            return signal

        return savgol_filter(signal, window_length=window, polyorder=3).tolist()

    def _dampen_untrusted_signal(self, samples):
        damped = []
        prev = float(samples[0].pos)

        for sample in samples:
            alpha = 1.0
            if sample.tracking_state == TrackingState.TRACKING_DEGRADED.value:
                alpha = 0.55
            elif sample.tracking_state in (
                TrackingState.PATTERN_BRIDGE.value,
                TrackingState.RECOVER_LOCAL.value,
                TrackingState.RECOVER_GLOBAL.value,
            ):
                alpha = 0.75

            if sample.provenance == SignalProvenance.REPLAYED_PATTERN.value:
                alpha = min(alpha, 0.70)
            elif sample.provenance in (
                SignalProvenance.RECOVERED_LOCAL.value,
                SignalProvenance.RECOVERED_GLOBAL.value,
            ):
                alpha = min(alpha, 0.82)

            if sample.confidence < 0.72:
                alpha *= max(0.20, sample.confidence / 0.72)

            prev = prev + (alpha * (float(sample.pos) - prev))
            damped.append(prev)

        return damped

    def _normalize_signal(self, signal, samples):
        trusted = [
            signal[i]
            for i, sample in enumerate(samples)
            if sample.tracking_state == TrackingState.TRACKING_CONFIDENT.value and sample.confidence >= 0.78
        ]
        base = trusted if len(trusted) >= 12 else signal

        low = float(np.percentile(base, 8))
        high = float(np.percentile(base, 92))
        if high - low < 12.0:
            low = float(np.percentile(signal, 4))
            high = float(np.percentile(signal, 96))

        if high - low < 8.0:
            raw_min = float(min(signal))
            raw_max = float(max(signal))
            if raw_max - raw_min < 1e-6:
                return [50 for _ in signal]
            low = raw_min
            high = raw_max

        normalized = []
        for value in signal:
            scaled = (float(value) - low) / max(1e-6, high - low)
            normalized.append(self._clamp_int(100.0 * max(0.0, min(1.0, scaled))))

        return normalized

    def _rebalance_envelope(self, signal):
        if len(signal) < 5:
            return signal

        p05 = float(np.percentile(signal, 5))
        p10 = float(np.percentile(signal, 10))
        p50 = float(np.percentile(signal, 50))
        p95 = float(np.percentile(signal, 95))
        mean_value = float(np.mean(signal))

        center_shift = min(
            12.0,
            (max(0.0, p10 - 8.0) * 0.80)
            + (max(0.0, mean_value - 47.0) * 0.45),
        )
        target_center = max(38.0, min(50.0, p50 - center_shift))

        lower_span = max(10.0, p50 - p05)
        upper_span = max(10.0, p95 - p50)
        lower_gamma = max(0.72, 0.90 - min(0.18, max(0.0, p10 - 8.0) / 40.0))
        upper_gamma = 1.04

        adjusted = []
        for value in signal:
            value = float(value)
            if value <= p50:
                ratio = max(0.0, min(1.0, (p50 - value) / lower_span))
                remapped = target_center - ((ratio**lower_gamma) * target_center)
            else:
                ratio = max(0.0, min(1.0, (value - p50) / upper_span))
                remapped = target_center + ((ratio**upper_gamma) * (100.0 - target_center))
            adjusted.append(self._clamp_int(remapped))

        return adjusted

    def _extract_peak_valley_actions(self, signal, samples):
        if not signal:
            return []

        amplitude_span = max(signal) - min(signal)
        hysteresis = max(6.0, min(12.0, amplitude_span * 0.08))
        min_gap_ms = max(48, int(round(1000.0 / 12.0)))

        base_idx = 0
        base_pos = float(signal[0])
        candidate_idx = 0
        candidate_pos = float(signal[0])
        trend = 0
        extrema = []

        for idx in range(1, len(signal)):
            pos = float(signal[idx])

            if trend == 0:
                delta = pos - base_pos
                if delta >= hysteresis:
                    extrema.append(base_idx)
                    trend = 1
                    candidate_idx = idx
                    candidate_pos = pos
                elif delta <= -hysteresis:
                    extrema.append(base_idx)
                    trend = -1
                    candidate_idx = idx
                    candidate_pos = pos
                elif abs(pos - 50.0) > abs(base_pos - 50.0):
                    base_idx = idx
                    base_pos = pos
            elif trend > 0:
                if pos >= candidate_pos:
                    candidate_idx = idx
                    candidate_pos = pos
                elif candidate_pos - pos >= hysteresis and samples[idx].at_ms - samples[candidate_idx].at_ms >= min_gap_ms:
                    extrema.append(candidate_idx)
                    trend = -1
                    candidate_idx = idx
                    candidate_pos = pos
            else:
                if pos <= candidate_pos:
                    candidate_idx = idx
                    candidate_pos = pos
                elif pos - candidate_pos >= hysteresis and samples[idx].at_ms - samples[candidate_idx].at_ms >= min_gap_ms:
                    extrema.append(candidate_idx)
                    trend = 1
                    candidate_idx = idx
                    candidate_pos = pos

        if extrema and extrema[-1] != candidate_idx:
            extrema.append(candidate_idx)

        unique_indices = []
        seen = set()
        for idx in extrema:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        actions = []
        for idx in unique_indices:
            actions.append({"at": int(samples[idx].at_ms), "pos": self._clamp_int(signal[idx])})

        return actions

    @staticmethod
    def _drop_redundant_actions(actions):
        if len(actions) < 3:
            return actions

        filtered = [actions[0]]
        for action in actions[1:]:
            last = filtered[-1]
            if action["at"] == last["at"]:
                if abs(action["pos"] - 50) >= abs(last["pos"] - 50):
                    filtered[-1] = action
                continue

            if abs(action["pos"] - last["pos"]) < 4:
                if abs(action["pos"] - 50) >= abs(last["pos"] - 50):
                    filtered[-1] = action
                continue

            filtered.append(action)

        if len(filtered) >= 3 and filtered[-1]["pos"] == filtered[-2]["pos"]:
            filtered.pop()

        return filtered

    def _snap_start_phase(self, actions, signal, samples):
        if len(actions) < 2 or len(signal) < 5:
            return actions

        start_at = int(samples[0].at_ms)
        horizon_ms = max(280, int(round(1000.0 / self.fps * 18.0)))
        early_indices = [idx for idx, sample in enumerate(samples) if sample.at_ms - start_at <= horizon_ms]
        if not early_indices:
            return actions

        early_values = [signal[idx] for idx in early_indices]
        early_peak = self._clamp_int(max(early_values))
        early_valley = self._clamp_int(min(early_values))
        early_mean = float(np.mean(early_values))

        strong_actions = [action for action in actions[:4] if abs(action["pos"] - 50) >= 22]
        if len(strong_actions) < 2:
            return actions

        first_strong = strong_actions[0]
        second_strong = strong_actions[1]
        start_action = None

        if first_strong["pos"] <= 35 and second_strong["pos"] >= 65:
            start_action = {"at": start_at, "pos": self._clamp_int(max(early_peak, second_strong["pos"]))}
        elif first_strong["pos"] >= 65 and second_strong["pos"] <= 35:
            start_action = {"at": start_at, "pos": self._clamp_int(min(early_valley, second_strong["pos"]))}
        elif early_mean >= 58.0 and first_strong["pos"] <= 35:
            start_action = {"at": start_at, "pos": self._clamp_int(max(early_peak, 84))}
        elif early_mean <= 42.0 and first_strong["pos"] >= 65:
            start_action = {"at": start_at, "pos": self._clamp_int(min(early_valley, 16))}

        if start_action is None:
            return actions

        snapped = list(actions)
        if snapped[0]["at"] <= start_at + max(34, int(round(1000.0 / self.fps))):
            if abs(start_action["pos"] - 50) > abs(snapped[0]["pos"] - 50):
                snapped[0] = start_action
        else:
            snapped.insert(0, start_action)

        return self._drop_redundant_actions(snapped)

    def generate(self, samples):
        if not samples:
            return []

        if len(samples) == 1:
            return [{"at": int(samples[0].at_ms), "pos": self._clamp_int(samples[0].pos)}]

        damped = self._dampen_untrusted_signal(samples)
        smoothed = self._smooth_signal(damped)
        normalized = self._normalize_signal(smoothed, samples)
        normalized = self._rebalance_envelope(normalized)
        actions = self._extract_peak_valley_actions(normalized, samples)
        actions = self._drop_redundant_actions(actions)
        actions = self._snap_start_phase(actions, normalized, samples)
        actions = self._drop_redundant_actions(actions)

        if len(actions) >= 2:
            return actions

        start = {"at": int(samples[0].at_ms), "pos": self._clamp_int(normalized[0])}
        end = {"at": int(samples[-1].at_ms), "pos": self._clamp_int(normalized[-1])}
        if start["at"] == end["at"] and start["pos"] == end["pos"]:
            return [start]
        return [start, end]
