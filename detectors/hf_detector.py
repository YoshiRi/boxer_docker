"""HuggingFace zero-shot detector adapter for Boxer.

Supports Grounding DINO and OWLv2 (HF).
Implements the same forward() interface as owl.OwlWrapper for drop-in use.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _per_class_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: list[int], iou_threshold: float) -> list[int]:
    """Per-class greedy NMS. boxes: (N,4) x1y1x2y2."""
    label_tensor = torch.tensor(labels, dtype=torch.long)
    keep = []
    for cls in label_tensor.unique():
        cls_mask = label_tensor == cls
        cls_idx = cls_mask.nonzero(as_tuple=True)[0]
        cls_scores = scores[cls_idx]
        cls_boxes = boxes[cls_idx]
        order = cls_scores.argsort(descending=True)
        cls_idx = cls_idx[order]
        cls_boxes = cls_boxes[order]
        suppressed = torch.zeros(len(cls_idx), dtype=torch.bool)
        for i in range(len(cls_idx)):
            if suppressed[i]:
                continue
            keep.append(int(cls_idx[i].item()))
            ix1 = torch.max(cls_boxes[i, 0], cls_boxes[i + 1:, 0])
            iy1 = torch.max(cls_boxes[i, 1], cls_boxes[i + 1:, 1])
            ix2 = torch.min(cls_boxes[i, 2], cls_boxes[i + 1:, 2])
            iy2 = torch.min(cls_boxes[i, 3], cls_boxes[i + 1:, 3])
            inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_j = (cls_boxes[i + 1:, 2] - cls_boxes[i + 1:, 0]) * (cls_boxes[i + 1:, 3] - cls_boxes[i + 1:, 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            suppressed[i + 1:] |= iou > iou_threshold
    keep.sort()
    return keep


class HFDetector:
    """HuggingFace zero-shot 2D detector.

    Drop-in replacement for owl.OwlWrapper — same forward() signature.

    Supported model families:
    - Grounding DINO: "IDEA-Research/grounding-dino-base" (default)
    - OWLv2 (HF):     "google/owlv2-base-patch16-ensemble"

    The forward() output uses Boxer's (x1, x2, y1, y2) box convention.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cpu",
        text_prompts: list[str] | None = None,
        min_confidence: float = 0.25,
        nms_iou_threshold: float = 0.5,
    ):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HFDetector. "
                "Install it with: pip install transformers timm"
            ) from exc

        self.model_id = model_id
        self.device = device
        self.text_prompts: list[str] = list(text_prompts or [])
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold

        # Detect model family from ID
        mid_lower = model_id.lower()
        self._is_owlv2 = "owl" in mid_lower
        self._is_gdino = "grounding-dino" in mid_lower or "groundingdino" in mid_lower

        print(f"Loading HF detector '{model_id}' on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        self._gdino_text: str = ""
        if self._is_gdino:
            self._build_gdino_text()

        print(
            f"Loaded HF detector '{model_id.split('/')[-1]}' "
            f"with {len(self.text_prompts)} prompts on {device}"
        )

    def _build_gdino_text(self) -> None:
        """Grounding DINO expects prompts joined as 'chair . table . sofa .'"""
        self._gdino_text = " . ".join(p.lower().strip() for p in self.text_prompts)
        if self._gdino_text:
            self._gdino_text += " ."

    def set_text_prompts(self, prompts: list[str]) -> None:
        self.text_prompts = list(prompts)
        if self._is_gdino:
            self._build_gdino_text()

    # ------------------------------------------------------------------
    # Public interface — matches owl.OwlWrapper.forward()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        image_torch: torch.Tensor,
        rotated: bool = False,
        resize_to_HW: tuple[int, int] = (960, 960),
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], None]:
        """Detect objects in image_torch.

        Args:
            image_torch: (1, C, H, W) float in [0, 255]
            rotated: if True, rotate 90° CW before detection and undo afterwards
            resize_to_HW: unused (HF models handle their own resizing)

        Returns:
            boxes:       (N, 4) in Boxer (x1, x2, y1, y2) pixel coords
            scores:      (N,) float32
            label_ints:  list[int] indices into self.text_prompts
            None
        """
        empty = torch.zeros((0, 4)), torch.zeros(0), [], None

        if not self.text_prompts:
            return empty

        # Convert tensor → uint8 numpy RGB
        img_np = image_torch[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if rotated:
            img_np = np.rot90(img_np, k=1).copy()  # 90° CCW = 3× CW
        H, W = img_np.shape[:2]

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_np)

        if self._is_owlv2:
            boxes, scores, label_ints = self._detect_owlv2(pil_img, H, W)
        else:
            boxes, scores, label_ints = self._detect_gdino(pil_img, H, W)

        if len(boxes) == 0:
            return empty

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        scores_t = torch.as_tensor(scores, dtype=torch.float32)

        # Size filter
        widths = boxes_t[:, 2] - boxes_t[:, 0]
        heights = boxes_t[:, 3] - boxes_t[:, 1]
        valid = (widths > 0.05 * W) & (heights > 0.05 * H) & (widths < 0.9 * W) & (heights < 0.9 * H)
        if not valid.any():
            return empty
        boxes_t = boxes_t[valid]
        scores_t = scores_t[valid]
        label_ints = [label_ints[i] for i in range(len(valid)) if valid[i]]

        # Per-class NMS (in xyxy for IoU calculation)
        if self.nms_iou_threshold < 1.0 and len(boxes_t) > 1:
            keep = _per_class_nms(boxes_t, scores_t, label_ints, self.nms_iou_threshold)
            boxes_t = boxes_t[keep]
            scores_t = scores_t[keep]
            label_ints = [label_ints[i] for i in keep]

        if len(boxes_t) == 0:
            return empty

        # Convert standard xyxy → Boxer convention (x1, x2, y1, y2)
        boxes_boxer = boxes_t[:, [0, 2, 1, 3]]

        if rotated:
            x1, x2, y1, y2 = boxes_boxer.unbind(-1)
            # Undo 90° CCW: swap and flip back
            orig_W = H  # after 90° CCW rotation, original W is now H
            boxes_boxer = torch.stack([y1, y2, orig_W - x2, orig_W - x1], dim=-1)

        return boxes_boxer, scores_t, label_ints, None

    # ------------------------------------------------------------------
    # Model-specific backends
    # ------------------------------------------------------------------

    def _detect_owlv2(
        self, pil_img, H: int, W: int
    ) -> tuple[list, list, list[int]]:
        inputs = self.processor(
            text=[self.text_prompts],
            images=pil_img,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.min_confidence, target_sizes=target_sizes
        )[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        label_ints = results["labels"].cpu().tolist()
        return boxes, scores, label_ints

    def _detect_gdino(
        self, pil_img, H: int, W: int
    ) -> tuple[list, list, list[int]]:
        if not self._gdino_text:
            return [], [], []
        inputs = self.processor(
            images=pil_img,
            text=self._gdino_text,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.min_confidence,
            text_threshold=max(0.1, self.min_confidence * 0.8),
            target_sizes=[(H, W)],
        )[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        phrases = results["labels"]
        label_ints = [self._match_phrase(p) for p in phrases]
        # Drop detections that could not be mapped
        valid_idx = [i for i, li in enumerate(label_ints) if li >= 0]
        return (
            [boxes[i] for i in valid_idx],
            [scores[i] for i in valid_idx],
            [label_ints[i] for i in valid_idx],
        )

    def _match_phrase(self, phrase: str) -> int:
        """Map a GDINO phrase back to the prompt list index."""
        phrase_lower = phrase.lower().strip()
        # Exact match
        for i, p in enumerate(self.text_prompts):
            if phrase_lower == p.lower().strip():
                return i
        # Substring match (phrase contained in prompt or vice versa)
        for i, p in enumerate(self.text_prompts):
            pl = p.lower().strip()
            if phrase_lower in pl or pl in phrase_lower:
                return i
        return -1
