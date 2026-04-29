import json
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class FunscriptValidator:
    def __init__(self):
        pass

    def calculate_score(self, generated_data, ground_truth_path):
        """
        Calculates DTW-based accuracy score compared to ground truth.
        """
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            # Extract positions and ensure they are 1-D arrays
            gen_pos = np.array([a['pos'] for a in generated_data['actions']], dtype=np.float64).flatten()
            gt_pos = np.array([a['pos'] for a in gt_data['actions']], dtype=np.float64).flatten()
            
            if len(gen_pos) == 0 or len(gt_pos) == 0:
                return {"accuracy": 0, "error": "Empty data"}

            # fastdtw with euclidean distance expects (N, D) arrays. 
            # For 1D scalar data, D=1.
            gen_pos = gen_pos.reshape(-1, 1)
            gt_pos = gt_pos.reshape(-1, 1)

            # DTW handles different lengths
            distance, path = fastdtw(gen_pos, gt_pos, dist=euclidean)
            
            # Normalize distance to a 0-100 score (lower distance = higher score)
            # max_dist: max possible distance (length * 100)
            max_dist = max(len(gen_pos), len(gt_pos)) * 100
            score = max(0, 100 * (1 - (distance / max_dist)))
            
            return {
                "dtw_distance": float(distance),
                "accuracy": round(float(score), 2)
            }
        except Exception as e:
            print(f"Validation Error: {e}")
            return {"error": str(e)}
