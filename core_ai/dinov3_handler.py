import os

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class DINOv3Handler:
    def __init__(self, model_path="Models/dinov3-vitl16-pretrain-lvd1689m.safetensors"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.input_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            raise FileNotFoundError(f"DINOv3 model not found at {self.model_path}")

    def load_model(self):
        try:
            print(f"Loading DINOv3 (ViT-L/16) local weights from {self.model_path}...")
            try:
                self.model = timm.create_model("vit_large_patch16_dinov3", pretrained=False)

                raw_state_dict = load_file(self.model_path)
                state_dict = {}

                if "model" in raw_state_dict:
                    raw_state_dict = raw_state_dict["model"]

                qkv_groups = {}
                for k, v in raw_state_dict.items():
                    new_k = k
                    if k == "embeddings.cls_token":
                        new_k = "cls_token"
                    elif k.startswith("embeddings.patch_embeddings."):
                        new_k = k.replace("embeddings.patch_embeddings.", "patch_embed.proj.")
                    elif k.startswith("layer."):
                        new_k = new_k.replace("layer.", "blocks.")
                        new_k = new_k.replace(".layer_scale1.lambda1", ".gamma_1")
                        new_k = new_k.replace(".layer_scale2.lambda1", ".gamma_2")
                        if ".attention." in new_k and any(x in new_k for x in [".q_proj.weight", ".k_proj.weight", ".v_proj.weight"]):
                            prefix = new_k.rsplit(".attention.", 1)[0]
                            group_key = f"{prefix}.attn.qkv.weight"
                            if group_key not in qkv_groups:
                                qkv_groups[group_key] = {"meta_prefix": f"{prefix.replace('blocks.', 'layer.')}.attention"}
                            continue
                        new_k = new_k.replace(".attention.o_proj.", ".attn.proj.")
                        new_k = new_k.replace(".mlp.up_proj.", ".mlp.fc1.")
                        new_k = new_k.replace(".mlp.down_proj.", ".mlp.fc2.")

                    if "mask_token" in new_k or "register_tokens" in new_k:
                        continue

                    state_dict[new_k] = v

                for group_key, info in qkv_groups.items():
                    meta_prefix = info["meta_prefix"]
                    try:
                        q = raw_state_dict[f"{meta_prefix}.q_proj.weight"]
                        k = raw_state_dict[f"{meta_prefix}.k_proj.weight"]
                        v = raw_state_dict[f"{meta_prefix}.v_proj.weight"]
                        state_dict[group_key] = torch.cat([q, k, v], dim=0)
                    except KeyError as e:
                        print(f"Skipping QKV weight concatenation {group_key}: missing {e}")

                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f"DINOv3 load status: {msg}")

                self.model.to(self.device).eval()
                self.mean = self.mean.to(self.device)
                self.std = self.std.to(self.device)
                if self.device.type == "cuda":
                    self.model.half()
                    self.mean = self.mean.half()
                    self.std = self.std.half()
                print(f"DINOv3 initialized on {self.device.type.upper()}.")
            except Exception as timm_e:
                print(f"Timm initialization failed: {timm_e}")
                self.model = None
        except Exception as e:
            print(f"Error loading DINOv3: {e}")
            self.model = None

    def _prepare_tensor(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()
        tensor = (tensor - self.mean) / self.std
        return tensor

    @staticmethod
    def _compute_centroid(mask, fallback_point, frame_shape):
        if mask is not None:
            moments = cv2.moments(mask)
            if moments["m00"] > 1e-5:
                c_x = int(moments["m10"] / moments["m00"])
                c_y = int(moments["m01"] / moments["m00"])
                return (c_x, c_y)

        if fallback_point is not None:
            return fallback_point

        h, w = frame_shape[:2]
        return (w // 2, h // 2)

    @staticmethod
    def _compute_crop_box(frame_shape, mask=None, prompt_point=None):
        h, w = frame_shape[:2]

        if mask is not None and np.count_nonzero(mask) > 0:
            x, y, bw, bh = cv2.boundingRect(mask)
            pad = int(max(bw, bh) * 0.35) + 12
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
        else:
            if prompt_point is None:
                prompt_point = (w // 2, h // 2)
            half = min(max(64, min(w, h) // 7), 128)
            x0 = max(0, prompt_point[0] - half)
            y0 = max(0, prompt_point[1] - half)
            x1 = min(w, prompt_point[0] + half)
            y1 = min(h, prompt_point[1] + half)

        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return (x0, y0, x1, y1)

    def _extract_embedding(self, crop_bgr):
        if self.model is None or crop_bgr is None or crop_bgr.size == 0:
            return None

        with torch.no_grad():
            tensor = self._prepare_tensor(crop_bgr)
            tokens = self.model.forward_features(tensor)
            if isinstance(tokens, dict):
                if "x_norm_clstoken" in tokens:
                    embedding = tokens["x_norm_clstoken"]
                elif "x_prenorm" in tokens:
                    embedding = tokens["x_prenorm"]
                else:
                    embedding = next((v for v in tokens.values() if isinstance(v, torch.Tensor)), None)
            else:
                embedding = tokens

            if embedding is None:
                return None

            if embedding.dim() == 3:
                cls_token = embedding[:, 0]
                patch_mean = embedding[:, 1:].mean(dim=1) if embedding.shape[1] > 1 else cls_token
                embedding = (cls_token + patch_mean) * 0.5
            elif embedding.dim() > 2:
                embedding = embedding.flatten(1)

            embedding = F.normalize(embedding.float(), dim=1)
            return embedding[0].detach().cpu().numpy().astype(np.float32)

    def extract_features(self, frame, mask=None, prompt_point=None):
        """
        Extract a target-region signature:
        - centroid for tracking prompt updates
        - appearance embedding for recovery / re-identification
        - coarse confidence score
        """
        try:
            centroid = self._compute_centroid(mask, prompt_point, frame.shape)
            crop_box = self._compute_crop_box(frame.shape, mask=mask, prompt_point=centroid)
            x0, y0, x1, y1 = crop_box
            crop = frame[y0:y1, x0:x1].copy()

            mask_fill_ratio = 0.0
            if mask is not None and crop.size > 0:
                crop_mask = mask[y0:y1, x0:x1]
                if crop_mask.size > 0:
                    mask_fill_ratio = float(np.count_nonzero(crop_mask)) / float(crop_mask.size)
                    if np.count_nonzero(crop_mask) > 0:
                        background = crop.copy()
                        background[crop_mask == 0] = (background[crop_mask == 0] * 0.18).astype(np.uint8)
                        crop = background

            embedding = self._extract_embedding(crop)
            confidence = 0.35 + (0.55 * min(1.0, mask_fill_ratio * 2.0))
            if embedding is not None:
                confidence += 0.10
            confidence = max(0.0, min(1.0, confidence))

            return {
                "centroid": centroid,
                "confidence": confidence,
                "embedding": embedding,
                "crop_box": crop_box,
                "point": prompt_point or centroid,
                "mask_fill_ratio": mask_fill_ratio,
            }
        except Exception as e:
            print(f"DINOv3 Inference Error: {e}")
            return {
                "centroid": self._compute_centroid(mask, prompt_point, frame.shape),
                "confidence": 0.0,
                "embedding": None,
                "crop_box": None,
                "point": prompt_point,
                "mask_fill_ratio": 0.0,
            }
