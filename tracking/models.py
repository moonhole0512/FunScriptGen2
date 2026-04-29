from dataclasses import asdict, dataclass
from enum import Enum


class TrackingState(str, Enum):
    INIT = "INIT"
    USER_CONFIRM = "USER_CONFIRM"
    TRACKING_CONFIDENT = "TRACKING_CONFIDENT"
    TRACKING_DEGRADED = "TRACKING_DEGRADED"
    RECOVER_LOCAL = "RECOVER_LOCAL"
    RECOVER_GLOBAL = "RECOVER_GLOBAL"
    PATTERN_BRIDGE = "PATTERN_BRIDGE"
    USER_REANNOTATE = "USER_REANNOTATE"
    ABORTED = "ABORTED"
    FINISHED = "FINISHED"


class SignalProvenance(str, Enum):
    TRACKED = "tracked"
    BRIDGED_SHORT_GAP = "bridged_short_gap"
    REPLAYED_PATTERN = "replayed_pattern"
    RECOVERED_LOCAL = "recovered_local"
    RECOVERED_GLOBAL = "recovered_global"
    USER_REANNOTATED = "user_reannotated"


@dataclass
class TrackingMetrics:
    mask_active: bool = False
    mask_area_ratio: float = 0.0
    mask_area_change_ratio: float = 1.0
    appearance_confidence: float = 0.0
    feature_count: int = 0
    flow_inlier_ratio: float = 0.0
    centroid_jump_px: float = 0.0
    boundary_pressure: float = 0.0
    prompt_locked: bool = False
    rhythm_range: float = 0.0
    ai_refresh_requested: bool = False
    bridge_ready: bool = False
    recovery_score: float = 0.0
    recovery_candidate_count: int = 0

    def as_dict(self):
        return asdict(self)


@dataclass
class TrackingConfidenceReport:
    overall_confidence: float
    mask_score: float
    mask_stability_score: float
    feature_score: float
    flow_score: float
    appearance_score: float
    centroid_score: float
    boundary_score: float
    rhythm_score: float

    def as_dict(self):
        return asdict(self)
