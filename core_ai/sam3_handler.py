import torch
import os
import cv2
import numpy as np
import sys
from pathlib import Path

_yolo_config_root = os.path.abspath(".cache")
os.makedirs(_yolo_config_root, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _yolo_config_root)

try:
    _path_exists = Path.exists

    def _safe_path_exists(self):
        try:
            return _path_exists(self)
        except PermissionError:
            return False

    Path.exists = _safe_path_exists
    from ultralytics import SAM
    SAM_IMPORT_ERROR = None
except Exception as import_error:
    SAM = None
    SAM_IMPORT_ERROR = import_error
finally:
    Path.exists = _path_exists

class SAM3Handler:
    def __init__(self, model_path="Models/sam3.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.abspath(model_path)
        self.model = None
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Warning: SAM3 model not found at {self.model_path}")

    def load_model(self):
        try:
            print(f"Initializing SAM3 (via Ultralytics) from {self.model_path}...")
            if SAM:
                # Ultralytics SAM handles Windows/Triton issues internally
                self.model = SAM(self.model_path)
                
                # OPTIMIZATION: Enable FP16 on GPU
                if self.device.type == 'cuda':
                    try:
                        self.model.model.half() # Apply FP16 to the underlying model
                        print(f"SAM3: FP16 (Half Precision) enabled on {self.device.type.upper()}.")
                    except Exception as fp_e:
                        print(f"SAM3: Could not enable FP16: {fp_e}")
                
                print(f"SAM3 (Ultralytics) initialized successfully on {self.device.type.upper()}.")
            else:
                print(f"Warning: 'ultralytics' package unavailable: {SAM_IMPORT_ERROR}")
                self.model = None
        except Exception as e:
            print(f"Error during SAM3 (Ultralytics) initialization: {e}")
            self.model = None

    def get_mask(self, frame, prompt_point=None):
        """
        Returns a binary mask (uint8) using SAM3 (Ultralytics).
        """
        if self.model is None or prompt_point is None:
            return None
            
        try:
            # Ultralytics SAM inference
            # points=[[x, y]], labels=[1] (foreground)
            # imgsz = 1036 to avoid "must be multiple of max stride 14" warning
            results = self.model.predict(
                frame, 
                points=[[prompt_point[0], prompt_point[1]]], 
                labels=[1], 
                device=self.device, 
                verbose=False,
                imgsz=1036
            )
            
            if results and len(results) > 0:
                if results[0].masks is not None and len(results[0].masks.data) > 0:
                    # Get the first mask (highest confidence usually)
                    mask = results[0].masks.data[0].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    
                    # Ensure mask matches original frame size
                    h, w = frame.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                    return mask
            return None
        except Exception as e:
            print(f"SAM3 (Ultralytics) Prediction Error: {e}")
            return None
