from dataclasses import dataclass

from .models import SignalProvenance, TrackingConfidenceReport, TrackingState


@dataclass
class TrackingStateSnapshot:
    state: TrackingState
    provenance: SignalProvenance
    degraded_streak: int
    recovery_streak: int
    broken_streak: int
    user_intervention_required: bool
    state_changed: bool


class TrackingStateMachine:
    def __init__(
        self,
        confident_threshold=0.80,
        degraded_threshold=0.55,
        recovery_threshold=0.35,
        pattern_bridge_budget_frames=24,
        user_reannotate_trigger_frames=72,
    ):
        self.confident_threshold = confident_threshold
        self.degraded_threshold = degraded_threshold
        self.recovery_threshold = recovery_threshold
        self.pattern_bridge_budget_frames = pattern_bridge_budget_frames
        self.user_reannotate_trigger_frames = user_reannotate_trigger_frames
        self.reset()

    def reset(self):
        self.state = TrackingState.INIT
        self.degraded_streak = 0
        self.recovery_streak = 0
        self.broken_streak = 0

    def advance(
        self,
        report: TrackingConfidenceReport,
        target_acquired: bool,
        can_pattern_bridge: bool,
    ) -> TrackingStateSnapshot:
        previous_state = self.state

        if not target_acquired:
            self.state = TrackingState.INIT
            self.degraded_streak = 0
            self.recovery_streak = 0
            self.broken_streak = 0
        else:
            score = report.overall_confidence
            if score >= self.confident_threshold:
                self.state = TrackingState.TRACKING_CONFIDENT
                self.degraded_streak = 0
                self.recovery_streak = 0
                self.broken_streak = 0
            elif score >= self.degraded_threshold:
                self.state = TrackingState.TRACKING_DEGRADED
                self.degraded_streak += 1
                self.recovery_streak = 0
                self.broken_streak = 0
            elif score >= self.recovery_threshold:
                self.degraded_streak += 1
                self.recovery_streak += 1
                self.broken_streak = 0
                if self.recovery_streak <= 12:
                    self.state = TrackingState.RECOVER_LOCAL
                else:
                    self.state = TrackingState.RECOVER_GLOBAL
            else:
                self.degraded_streak += 1
                self.recovery_streak += 1
                self.broken_streak += 1
                if can_pattern_bridge and self.broken_streak <= self.pattern_bridge_budget_frames:
                    self.state = TrackingState.PATTERN_BRIDGE
                elif self.broken_streak <= self.user_reannotate_trigger_frames:
                    self.state = TrackingState.RECOVER_GLOBAL
                else:
                    self.state = TrackingState.USER_REANNOTATE

        provenance = self._state_to_provenance(self.state)
        state_changed = self.state != previous_state

        return TrackingStateSnapshot(
            state=self.state,
            provenance=provenance,
            degraded_streak=self.degraded_streak,
            recovery_streak=self.recovery_streak,
            broken_streak=self.broken_streak,
            user_intervention_required=self.state == TrackingState.USER_REANNOTATE,
            state_changed=state_changed,
        )

    @staticmethod
    def _state_to_provenance(state: TrackingState) -> SignalProvenance:
        if state in (TrackingState.TRACKING_CONFIDENT, TrackingState.TRACKING_DEGRADED):
            return SignalProvenance.TRACKED
        if state == TrackingState.RECOVER_LOCAL:
            return SignalProvenance.RECOVERED_LOCAL
        if state == TrackingState.RECOVER_GLOBAL:
            return SignalProvenance.RECOVERED_GLOBAL
        if state == TrackingState.PATTERN_BRIDGE:
            return SignalProvenance.REPLAYED_PATTERN
        if state == TrackingState.USER_REANNOTATE:
            return SignalProvenance.USER_REANNOTATED
        return SignalProvenance.BRIDGED_SHORT_GAP
