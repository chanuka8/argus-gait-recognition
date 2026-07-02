import cv2
import numpy as np


class SilhouetteStep:
    def __init__(
        self,
        target_size: tuple[int, int] = (64, 128),
    ) -> None:
        self.target_size = target_size

    def extract_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None

        # Extract grayscale from person crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Apply blur
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        # Apply Otsu threshold
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Clean with morphology open/close
        mask = self._clean_mask(mask)

        # Detect contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        # Keep only largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Remove small noise / reject abnormal areas
        crop_area = mask.shape[0] * mask.shape[1]
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < 50 or contour_area > 0.95 * crop_area:
            return None

        # Crop largest contour bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w < 5 or h < 15:
            return None

        # Reject abnormal aspect ratio (height / width)
        aspect_ratio = h / w
        if aspect_ratio < 1.2 or aspect_ratio > 6.0:
            return None

        # Draw the largest contour on a clean mask to filter other objects
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cropped_silhouette = contour_mask[y:y+h, x:x+w]

        # Resize while preserving aspect ratio and normalize body height to approximately 85% of canvas height
        target_h = int(self.target_size[1] * 0.85)  # 108 pixels
        scale_factor = target_h / h
        new_w = int(w * scale_factor)
        new_w = max(1, min(new_w, self.target_size[0]))

        resized_silhouette = cv2.resize(
            cropped_silhouette,
            (new_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Place silhouette on fixed 64x128 black canvas
        canvas = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

        # Center horizontally
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - target_h) // 2

        # Draw onto canvas
        canvas[y_offset : y_offset + target_h, x_offset : x_offset + new_w] = resized_silhouette

        # Return uint8 binary silhouette with values 0 or 255
        return canvas

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones(
            (3, 3),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        return mask