"""
Plate Detection Service
Uses YOLO for plate detection and PaddleOCR for text extraction.
Optimized for Indian license plates.
MEMORY OPTIMIZED: Uses recognition-only mode for lower RAM usage
"""

import re
import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from app.config.config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE,
    OCR_CONFIDENCE_THRESHOLD,
    MIN_PLATE_WIDTH,
    MIN_PLATE_HEIGHT
)

# Suppress PaddleOCR verbose logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Optimization constants
MAX_IMAGE_DIMENSION = 1280
YOLO_INPUT_SIZE = 320
MAX_PLATE_HEIGHT = 150


@dataclass
class PlateDetectionResult:
    """Result of plate detection"""
    success: bool
    plate_number: Optional[str] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    raw_text: Optional[str] = None
    yolo_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    total_time_ms: float = 0.0


class PlateDetectionService:
    """
    Service for detecting and reading vehicle license plates.
    MEMORY OPTIMIZED: Uses TextRecognition module only (skips detection model)
    """
    
    _instance = None
    _model = None
    _text_recognizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PlateDetectionService._model is None:
            self._load_models()
    
    def _load_models(self):
        """Load YOLO and PaddleOCR TextRecognition models"""
        logger.info("Loading plate detection models...")
        
        # Load YOLO model
        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
        
        PlateDetectionService._model = YOLO(str(model_path))
        logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
        
        # Load PaddleOCR - Try recognition-only mode first for memory savings
        try:
            from paddlex import TextRecognition
            # Use mobile model for lower memory footprint
            PlateDetectionService._text_recognizer = TextRecognition()
            logger.info("PaddleOCR TextRecognition module loaded (memory optimized)")
            self._use_recognition_only = True
        except ImportError:
            # Fallback to full PaddleOCR if TextRecognition not available
            logger.warning("TextRecognition not available, using full PaddleOCR")
            from paddleocr import PaddleOCR
            PlateDetectionService._text_recognizer = PaddleOCR(
                lang='en',
                enable_mkldnn=False
            )
            self._use_recognition_only = False
            logger.info("PaddleOCR full pipeline loaded")
        
        # Warmup
        self._warmup_ocr()
    
    def _warmup_ocr(self):
        """Warmup OCR model with dummy prediction"""
        logger.info("Warming up OCR model...")
        try:
            dummy_img = np.zeros((50, 200, 3), dtype=np.uint8)
            cv2.rectangle(dummy_img, (10, 15), (190, 35), (255, 255, 255), -1)
            self._run_ocr(dummy_img)
            logger.info("OCR warmup complete")
        except Exception as e:
            logger.warning(f"OCR warmup failed (non-critical): {e}")
    
    def _run_ocr(self, image: np.ndarray) -> Tuple[list, list]:
        """Run OCR and return texts and confidences"""
        all_texts = []
        all_confidences = []
        
        try:
            if self._use_recognition_only:
                # TextRecognition mode - direct recognition
                result = PlateDetectionService._text_recognizer.predict(image)
                # Parse TextRecognition result
                if result and hasattr(result, '__iter__'):
                    for item in result:
                        if isinstance(item, dict):
                            text = item.get('rec_text', item.get('text', ''))
                            score = item.get('rec_score', item.get('score', 0.0))
                            if score > OCR_CONFIDENCE_THRESHOLD:
                                all_texts.append(str(text))
                                all_confidences.append(float(score))
            else:
                # Full PaddleOCR mode
                ocr_result = PlateDetectionService._text_recognizer.predict(image)
                all_texts, all_confidences = self._parse_paddleocr_result(ocr_result)
        except Exception as e:
            logger.warning(f"OCR execution warning: {e}")
        
        return all_texts, all_confidences
    
    def _parse_paddleocr_result(self, ocr_result) -> Tuple[list, list]:
        """Parse full PaddleOCR result format"""
        all_texts = []
        all_confidences = []
        
        if not ocr_result:
            return all_texts, all_confidences
        
        try:
            for item in ocr_result:
                if item is None:
                    continue
                if isinstance(item, dict):
                    texts = item.get('rec_texts', [])
                    scores = item.get('rec_scores', [])
                    for t, s in zip(texts, scores):
                        if s > OCR_CONFIDENCE_THRESHOLD:
                            all_texts.append(str(t))
                            all_confidences.append(float(s))
                elif isinstance(item, list):
                    for sub_item in item:
                        if sub_item is None or len(sub_item) < 2:
                            continue
                        text_conf = sub_item[-1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            text = text_conf[0]
                            conf = text_conf[1]
                            if isinstance(conf, (int, float)) and conf > OCR_CONFIDENCE_THRESHOLD:
                                all_texts.append(str(text))
                                all_confidences.append(float(conf))
        except Exception as e:
            logger.warning(f"OCR result parsing warning: {e}")
        
        return all_texts, all_confidences
    
    def validate_indian_plate(self, text: str) -> Optional[str]:
        """Validate and clean Indian license plate format."""
        if not text:
            return None
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', clean_text):
            return clean_text
        
        return None
    
    def _resize_if_large(self, image: np.ndarray) -> np.ndarray:
        """Resize large images to speed up processing"""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _preprocess_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for speed"""
        height, width = plate_crop.shape[:2]
        
        if height > MAX_PLATE_HEIGHT:
            scale = MAX_PLATE_HEIGHT / height
            new_width = int(width * scale)
            plate_crop = cv2.resize(plate_crop, (new_width, MAX_PLATE_HEIGHT), interpolation=cv2.INTER_AREA)
        
        return plate_crop
    
    def _extract_text_from_crop(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float, float]:
        """Extract text from plate crop"""
        processed = self._preprocess_plate_crop(plate_crop)
        
        ocr_start = time.perf_counter()
        all_texts, all_confidences = self._run_ocr(processed)
        ocr_time_ms = (time.perf_counter() - ocr_start) * 1000
        
        if not all_texts:
            return None, 0.0, ocr_time_ms
        
        # Try each text individually
        for text, conf in zip(all_texts, all_confidences):
            valid_plate = self.validate_indian_plate(text)
            if valid_plate:
                return valid_plate, conf, ocr_time_ms
        
        # Try joining all texts
        if len(all_texts) > 1:
            joined_text = "".join(all_texts)
            valid_plate = self.validate_indian_plate(joined_text)
            if valid_plate:
                avg_conf = sum(all_confidences) / len(all_confidences)
                return valid_plate, avg_conf, ocr_time_ms
        
        raw_text = "".join(all_texts)
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        return raw_text, avg_conf, ocr_time_ms
    
    def detect_plate(self, image_bytes: bytes) -> PlateDetectionResult:
        """Detect and read license plate from image bytes."""
        total_start = time.perf_counter()
        yolo_time_ms = 0.0
        ocr_time_ms = 0.0
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return PlateDetectionResult(
                    success=False,
                    error_message="Invalid image format. Please upload a valid image file."
                )
            
            image = self._resize_if_large(image)
            
            yolo_start = time.perf_counter()
            results = PlateDetectionService._model.predict(
                image,
                conf=DETECTION_CONFIDENCE,
                verbose=False,
                imgsz=YOLO_INPUT_SIZE
            )
            yolo_time_ms = (time.perf_counter() - yolo_start) * 1000
            
            best_plate = None
            best_confidence = 0.0
            raw_texts = []
            
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    
                    width = x2 - x1
                    height = y2 - y1
                    if width < MIN_PLATE_WIDTH or height < MIN_PLATE_HEIGHT:
                        continue
                    
                    plate_crop = image[max(0, y1):y2, max(0, x1):x2]
                    if plate_crop.size == 0:
                        continue
                    
                    text, confidence, plate_ocr_time = self._extract_text_from_crop(plate_crop)
                    ocr_time_ms += plate_ocr_time
                    
                    if text:
                        raw_texts.append(text)
                        
                        valid_plate = self.validate_indian_plate(text)
                        if valid_plate and confidence > best_confidence:
                            best_plate = valid_plate
                            best_confidence = confidence
            
            total_time_ms = (time.perf_counter() - total_start) * 1000
            
            logging.info(f"TIMING PROFILE | YOLO: {yolo_time_ms:.1f}ms | OCR: {ocr_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms")
            
            if best_plate:
                return PlateDetectionResult(
                    success=True,
                    plate_number=best_plate,
                    confidence=best_confidence,
                    yolo_time_ms=yolo_time_ms,
                    ocr_time_ms=ocr_time_ms,
                    total_time_ms=total_time_ms
                )
            elif raw_texts:
                return PlateDetectionResult(
                    success=False,
                    error_message="Could not read plate clearly. Please upload a clearer image.",
                    raw_text=", ".join(raw_texts),
                    confidence=best_confidence,
                    yolo_time_ms=yolo_time_ms,
                    ocr_time_ms=ocr_time_ms,
                    total_time_ms=total_time_ms
                )
            else:
                return PlateDetectionResult(
                    success=False,
                    error_message="No license plate detected in image. Please upload an image with a visible plate.",
                    yolo_time_ms=yolo_time_ms,
                    ocr_time_ms=ocr_time_ms,
                    total_time_ms=total_time_ms
                )
                
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            return PlateDetectionResult(
                success=False,
                error_message=f"Error processing image: {str(e)}"
            )


# Singleton instance
plate_detection_service = PlateDetectionService()
