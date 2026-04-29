import os
import datetime

class VersionLogger:
    def __init__(self, log_dir="version_log"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_run(self, version, tech_stack, changes, validation_scores, tracking_summary=None):
        """
        Generates a markdown log file for the run.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"v_{version}_log.md"
        filepath = os.path.join(self.log_dir, filename)

        tracking_section = ""
        if tracking_summary:
            tracking_section = f"""
## Tracking Summary
- **Trusted Tracking Ratio**: {tracking_summary.get('trusted_tracking_ratio', 'N/A')}
- **Pattern Bridge Ratio**: {tracking_summary.get('pattern_bridge_ratio', 'N/A')}
- **Local Recovery Ratio**: {tracking_summary.get('local_recovery_ratio', 'N/A')}
- **Global Recovery Ratio**: {tracking_summary.get('global_recovery_ratio', 'N/A')}
- **Manual Review Ratio**: {tracking_summary.get('manual_review_ratio', 'N/A')}
- **Stitch Transition Ratio**: {tracking_summary.get('stitch_transition_ratio', 'N/A')}
- **Stitch Transform Ratio**: {tracking_summary.get('stitch_transform_ratio', 'N/A')}
- **Mean Confidence**: {tracking_summary.get('mean_confidence', 'N/A')}
- **Mean Stitch Confidence**: {tracking_summary.get('mean_stitch_confidence', 'N/A')}
"""

        log_content = f"""# Version Log: {version}
- **Date**: {timestamp}
- **Tech Used**: {tech_stack}

## Changes
{changes}

## Validation Scores
- **MSE**: {validation_scores.get('mse', 'N/A')}
- **DTW**: {validation_scores.get('dtw_distance', validation_scores.get('dtw', 'N/A'))}
- **Accuracy**: {validation_scores.get('accuracy', 'N/A')}%
{tracking_section}

---
*Automated Log Generation*
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(log_content)
        return filepath
