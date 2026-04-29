from .models import TrackingConfidenceReport, TrackingMetrics


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


class TrackingConfidenceModel:
    def __init__(self):
        self.expected_feature_count = 24.0
        self.max_centroid_jump_px = 96.0
        self.target_rhythm_range = 20.0

    def evaluate(self, metrics: TrackingMetrics) -> TrackingConfidenceReport:
        mask_score = 1.0 if metrics.mask_active else 0.0

        # Stable masks should change gradually; large swings indicate drift.
        mask_stability_score = 1.0
        if metrics.mask_active:
            mask_stability_score = _clamp(1.0 - (metrics.mask_area_change_ratio / 1.5))

        feature_score = _clamp(metrics.feature_count / self.expected_feature_count)
        flow_score = _clamp(metrics.flow_inlier_ratio)
        appearance_score = _clamp(metrics.appearance_confidence)
        centroid_score = _clamp(1.0 - (metrics.centroid_jump_px / self.max_centroid_jump_px))
        boundary_score = _clamp(1.0 - metrics.boundary_pressure)
        rhythm_score = _clamp(metrics.rhythm_range / self.target_rhythm_range)

        weighted_total = (
            (0.18 * mask_score)
            + (0.16 * mask_stability_score)
            + (0.16 * feature_score)
            + (0.16 * flow_score)
            + (0.12 * appearance_score)
            + (0.10 * centroid_score)
            + (0.06 * boundary_score)
            + (0.06 * rhythm_score)
        )

        # Apply conservative caps when the tracker looks structurally broken.
        if not metrics.mask_active and metrics.feature_count == 0:
            weighted_total = min(weighted_total, 0.30)
        elif metrics.flow_inlier_ratio < 0.10 and metrics.feature_count < 4:
            weighted_total = min(weighted_total, 0.42)

        if metrics.ai_refresh_requested:
            weighted_total *= 0.96

        overall_confidence = _clamp(weighted_total)

        return TrackingConfidenceReport(
            overall_confidence=overall_confidence,
            mask_score=mask_score,
            mask_stability_score=mask_stability_score,
            feature_score=feature_score,
            flow_score=flow_score,
            appearance_score=appearance_score,
            centroid_score=centroid_score,
            boundary_score=boundary_score,
            rhythm_score=rhythm_score,
        )
