# """
# Plate Detection Service
# Uses YOLO for plate detection and PaddleOCR for text extraction.
# Optimized for Indian license plates.
# MEMORY OPTIMIZED: Uses recognition-only mode for lower RAM usage
# """

# import re
# import cv2
# import numpy as np
# import logging
# import time
# import psutil
# import os
# import tracemalloc
# from typing import Optional, Tuple
# from dataclasses import dataclass
# from pathlib import Path

# from ultralytics import YOLO

# from app.config.config import (
#     YOLO_MODEL_PATH,
#     DETECTION_CONFIDENCE,
#     OCR_CONFIDENCE_THRESHOLD,
#     MIN_PLATE_WIDTH,
#     MIN_PLATE_HEIGHT
# )

# # Suppress PaddleOCR verbose logging
# logging.getLogger("ppocr").setLevel(logging.ERROR)

# logger = logging.getLogger(__name__)

# # Optimization constants
# MAX_IMAGE_DIMENSION = 1280
# YOLO_INPUT_SIZE = 320
# MAX_PLATE_HEIGHT = 150


# @dataclass
# class PlateDetectionResult:
#     """Result of plate detection"""
#     success: bool
#     plate_number: Optional[str] = None
#     confidence: float = 0.0
#     error_message: Optional[str] = None
#     raw_text: Optional[str] = None
#     yolo_time_ms: float = 0.0
#     ocr_time_ms: float = 0.0
#     total_time_ms: float = 0.0


# def get_detailed_memory_usage() -> dict:
#     """Get detailed memory usage stats using psutil"""
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     virtual_mem = psutil.virtual_memory()
    
#     return {
#         'process_rss_mb': mem_info.rss / (1024 * 1024),
#         'process_vms_mb': mem_info.vms / (1024 * 1024),
#         'system_available_mb': virtual_mem.available / (1024 * 1024),
#         'system_total_mb': virtual_mem.total / (1024 * 1024),
#         'system_percent_used': virtual_mem.percent
#     }


# class PlateDetectionService:
#     """
#     Service for detecting and reading vehicle license plates.
#     MEMORY OPTIMIZED: Uses TextRecognition module only (skips detection model)
#     """
    
#     _instance = None
#     _model = None
#     _text_recognizer = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def __init__(self):
#         if PlateDetectionService._model is None:
#             self._load_models()
    
#     def _load_models(self):
#         """Load YOLO and PaddleOCR TextRecognition models"""
#         logger.info("Loading plate detection models...")
        
#         # Load YOLO model
#         model_path = Path(YOLO_MODEL_PATH)
#         if not model_path.exists():
#             raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
        
#         PlateDetectionService._model = YOLO(str(model_path))
#         logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
        
#         # Load PaddleOCR - Try recognition-only mode first for memory savings
#         try:
#             from paddlex import TextRecognition
#             # Use mobile model for lower memory footprint
#             PlateDetectionService._text_recognizer = TextRecognition()
#             logger.info("PaddleOCR TextRecognition module loaded (memory optimized)")
#             self._use_recognition_only = True
#         except ImportError:
#             # Fallback to full PaddleOCR if TextRecognition not available
#             logger.warning("TextRecognition not available, using full PaddleOCR")
#             from paddleocr import PaddleOCR
#             PlateDetectionService._text_recognizer = PaddleOCR(
#                 lang='en',
#                 enable_mkldnn=False
#             )
#             self._use_recognition_only = False
#             logger.info("PaddleOCR full pipeline loaded")
        
#         # Warmup
#         self._warmup_ocr()
    
#     def _warmup_ocr(self):
#         """Warmup OCR model with dummy prediction"""
#         logger.info("Warming up OCR model...")
#         try:
#             dummy_img = np.zeros((50, 200, 3), dtype=np.uint8)
#             cv2.rectangle(dummy_img, (10, 15), (190, 35), (255, 255, 255), -1)
#             self._run_ocr(dummy_img)
#             logger.info("OCR warmup complete")
#         except Exception as e:
#             logger.warning(f"OCR warmup failed (non-critical): {e}")
    
#     def _run_ocr(self, image: np.ndarray) -> Tuple[list, list]:
#         """Run OCR and return texts and confidences"""
#         all_texts = []
#         all_confidences = []
        
#         try:
#             if self._use_recognition_only:
#                 # TextRecognition mode - direct recognition
#                 result = PlateDetectionService._text_recognizer.predict(image)
#                 # Parse TextRecognition result
#                 if result and hasattr(result, '__iter__'):
#                     for item in result:
#                         if isinstance(item, dict):
#                             text = item.get('rec_text', item.get('text', ''))
#                             score = item.get('rec_score', item.get('score', 0.0))
#                             if score > OCR_CONFIDENCE_THRESHOLD:
#                                 all_texts.append(str(text))
#                                 all_confidences.append(float(score))
#             else:
#                 # Full PaddleOCR mode
#                 ocr_result = PlateDetectionService._text_recognizer.predict(image)
#                 all_texts, all_confidences = self._parse_paddleocr_result(ocr_result)
#         except Exception as e:
#             logger.warning(f"OCR execution warning: {e}")
        
#         return all_texts, all_confidences
    
#     def _parse_paddleocr_result(self, ocr_result) -> Tuple[list, list]:
#         """Parse full PaddleOCR result format"""
#         all_texts = []
#         all_confidences = []
        
#         if not ocr_result:
#             return all_texts, all_confidences
        
#         try:
#             for item in ocr_result:
#                 if item is None:
#                     continue
#                 if isinstance(item, dict):
#                     texts = item.get('rec_texts', [])
#                     scores = item.get('rec_scores', [])
#                     for t, s in zip(texts, scores):
#                         if s > OCR_CONFIDENCE_THRESHOLD:
#                             all_texts.append(str(t))
#                             all_confidences.append(float(s))
#                 elif isinstance(item, list):
#                     for sub_item in item:
#                         if sub_item is None or len(sub_item) < 2:
#                             continue
#                         text_conf = sub_item[-1]
#                         if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
#                             text = text_conf[0]
#                             conf = text_conf[1]
#                             if isinstance(conf, (int, float)) and conf > OCR_CONFIDENCE_THRESHOLD:
#                                 all_texts.append(str(text))
#                                 all_confidences.append(float(conf))
#         except Exception as e:
#             logger.warning(f"OCR result parsing warning: {e}")
        
#         return all_texts, all_confidences
    
#     def validate_indian_plate(self, text: str) -> Optional[str]:
#         """Validate and clean Indian license plate format."""
#         if not text:
#             return None
        
#         clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
#         if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', clean_text):
#             return clean_text
        
#         return None
    
#     def _resize_if_large(self, image: np.ndarray) -> np.ndarray:
#         """Resize large images to speed up processing"""
#         height, width = image.shape[:2]
#         max_dim = max(height, width)
        
#         if max_dim > MAX_IMAGE_DIMENSION:
#             scale = MAX_IMAGE_DIMENSION / max_dim
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
#         return image
    
#     def _preprocess_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
#         """Minimal preprocessing for speed"""
#         height, width = plate_crop.shape[:2]
        
#         if height > MAX_PLATE_HEIGHT:
#             scale = MAX_PLATE_HEIGHT / height
#             new_width = int(width * scale)
#             plate_crop = cv2.resize(plate_crop, (new_width, MAX_PLATE_HEIGHT), interpolation=cv2.INTER_AREA)
        
#         return plate_crop
    
#     def _extract_text_from_crop(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float, float]:
#         """Extract text from plate crop"""
#         processed = self._preprocess_plate_crop(plate_crop)
        
#         ocr_start = time.perf_counter()
#         all_texts, all_confidences = self._run_ocr(processed)
#         ocr_time_ms = (time.perf_counter() - ocr_start) * 1000
        
#         if not all_texts:
#             return None, 0.0, ocr_time_ms
        
#         # Try each text individually
#         for text, conf in zip(all_texts, all_confidences):
#             valid_plate = self.validate_indian_plate(text)
#             if valid_plate:
#                 return valid_plate, conf, ocr_time_ms
        
#         # Try joining all texts
#         if len(all_texts) > 1:
#             joined_text = "".join(all_texts)
#             valid_plate = self.validate_indian_plate(joined_text)
#             if valid_plate:
#                 avg_conf = sum(all_confidences) / len(all_confidences)
#                 return valid_plate, avg_conf, ocr_time_ms
        
#         raw_text = "".join(all_texts)
#         avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
#         return raw_text, avg_conf, ocr_time_ms
    
#     def detect_plate(self, image_bytes: bytes) -> PlateDetectionResult:
#         """Detect and read license plate from image bytes."""
#         total_start = time.perf_counter()
#         yolo_time_ms = 0.0
#         ocr_time_ms = 0.0
        
#         # Start memory tracking
#         tracemalloc.start()
#         mem_before = get_detailed_memory_usage()
        
#         try:
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if image is None:
#                 return PlateDetectionResult(
#                     success=False,
#                     error_message="Invalid image format. Please upload a valid image file."
#                 )
            
#             image = self._resize_if_large(image)
            
#             yolo_start = time.perf_counter()
#             results = PlateDetectionService._model.predict(
#                 image,
#                 conf=DETECTION_CONFIDENCE,
#                 verbose=False,
#                 imgsz=YOLO_INPUT_SIZE
#             )
#             yolo_time_ms = (time.perf_counter() - yolo_start) * 1000
            
#             best_plate = None
#             best_confidence = 0.0
#             raw_texts = []
            
#             for result in results:
#                 for box in result.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box)
                    
#                     width = x2 - x1
#                     height = y2 - y1
#                     if width < MIN_PLATE_WIDTH or height < MIN_PLATE_HEIGHT:
#                         continue
                    
#                     plate_crop = image[max(0, y1):y2, max(0, x1):x2]
#                     if plate_crop.size == 0:
#                         continue
                    
#                     text, confidence, plate_ocr_time = self._extract_text_from_crop(plate_crop)
#                     ocr_time_ms += plate_ocr_time
                    
#                     if text:
#                         raw_texts.append(text)
                        
#                         valid_plate = self.validate_indian_plate(text)
#                         if valid_plate and confidence > best_confidence:
#                             best_plate = valid_plate
#                             best_confidence = confidence
            
#             total_time_ms = (time.perf_counter() - total_start) * 1000
            
#             # Get memory stats
#             mem_after = get_detailed_memory_usage()
#             current_mem, peak_mem = tracemalloc.get_traced_memory()
#             tracemalloc.stop()
            
#             peak_delta_mb = peak_mem / (1024 * 1024)
#             process_mem_mb = mem_after['process_rss_mb']
#             available_mb = mem_after['system_available_mb']
            
#             logging.info(
#                 f"TIMING PROFILE | YOLO: {yolo_time_ms:.1f}ms | OCR: {ocr_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms | "
#                 f"RAM: {process_mem_mb:.1f}MB | Peak Delta: {peak_delta_mb:.1f}MB | Available: {available_mb:.0f}MB"
#             )
            
#             if best_plate:
#                 return PlateDetectionResult(
#                     success=True,
#                     plate_number=best_plate,
#                     confidence=best_confidence,
#                     yolo_time_ms=yolo_time_ms,
#                     ocr_time_ms=ocr_time_ms,
#                     total_time_ms=total_time_ms
#                 )
#             elif raw_texts:
#                 return PlateDetectionResult(
#                     success=False,
#                     error_message="Could not read plate clearly. Please upload a clearer image.",
#                     raw_text=", ".join(raw_texts),
#                     confidence=best_confidence,
#                     yolo_time_ms=yolo_time_ms,
#                     ocr_time_ms=ocr_time_ms,
#                     total_time_ms=total_time_ms
#                 )
#             else:
#                 return PlateDetectionResult(
#                     success=False,
#                     error_message="No license plate detected in image. Please upload an image with a visible plate.",
#                     yolo_time_ms=yolo_time_ms,
#                     ocr_time_ms=ocr_time_ms,
#                     total_time_ms=total_time_ms
#                 )
                
#         except Exception as e:
#             logger.error(f"Plate detection error: {e}")
#             return PlateDetectionResult(
#                 success=False,
#                 error_message=f"Error processing image: {str(e)}"
#             )


# # Singleton instance
# plate_detection_service = PlateDetectionService()




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
import psutil
import os

import base64
import requests
import json
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from app.config.config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE,
    OCR_CONFIDENCE_THRESHOLD,
    MIN_PLATE_WIDTH,
    MIN_PLATE_HEIGHT,
    NVIDIA_OCR_API_KEY,
    NVIDIA_OCR_API_URL
)


import logging
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


def get_detailed_memory_usage() -> dict:
    """Get detailed memory usage stats using psutil"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return {
        'process_rss_mb': mem_info.rss / (1024 * 1024),
        'process_vms_mb': mem_info.vms / (1024 * 1024),
        'system_available_mb': virtual_mem.available / (1024 * 1024),
        'system_total_mb': virtual_mem.total / (1024 * 1024),
        'system_percent_used': virtual_mem.percent
    }


class PlateDetectionService:
    """
    Service for detecting and reading vehicle license plates.
    MEMORY OPTIMIZED: Uses recognition-only mode (skips detection model)
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
        """Load YOLO model (OCR is handled via NVIDIA API)"""
        logger.info("Loading plate detection models...")
        
        # Load YOLO model
        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
        
        PlateDetectionService._model = YOLO(str(model_path))
        logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
        logger.info("OCR configured to use NVIDIA API")
        
        # Log startup baseline RAM
        mem = get_detailed_memory_usage()
        logging.info(
            f"STARTUP RAM | Process RSS: {mem['process_rss_mb']:.1f}MB | "
            f"Process VMS: {mem['process_vms_mb']:.1f}MB | "
            f"System Available: {mem['system_available_mb']:.0f}MB / {mem['system_total_mb']:.0f}MB | "
            f"System Used: {mem['system_percent_used']:.1f}%"
        )
        print(
            f"STARTUP RAM | Process RSS: {mem['process_rss_mb']:.1f}MB | "
            f"Process VMS: {mem['process_vms_mb']:.1f}MB | "
            f"System Available: {mem['system_available_mb']:.0f}MB / {mem['system_total_mb']:.0f}MB | "
            f"System Used: {mem['system_percent_used']:.1f}%"
        )

    
    def _run_ocr(self, image: np.ndarray) -> Tuple[list, list]:
        """Run OCR via NVIDIA API and return texts and confidences"""
        all_texts = []
        all_confidences = []
        
        try:
            # Encode image to base64
            success, encoded_img = cv2.imencode('.jpg', image)
            if not success:
                logger.error("Failed to encode image for OCR API")
                return [], []
            
            image_b64 = base64.b64encode(encoded_img.tobytes()).decode()
            
            # Check payload size (limit is approx 180KB but safest to be under)
            if len(image_b64) > 180000:
                logger.warning(f"Image too large for NVIDIA API ({len(image_b64)} chars). resizing...")
                # Resize logic could go here if needed, but for plate crops it should be fine
                # Attempting aggressive resize if too big
                h, w = image.shape[:2]
                scale = 0.5
                image = cv2.resize(image, (int(w*scale), int(h*scale)))
                success, encoded_img = cv2.imencode('.jpg', image)
                image_b64 = base64.b64encode(encoded_img.tobytes()).decode()
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_OCR_API_KEY}",
                "Accept": "application/json"
            }
            
            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
            
            response = requests.post(NVIDIA_OCR_API_URL, headers=headers, json=payload, timeout=5)
            
            if response.status_code != 200:
                logger.error(f"NVIDIA API Error: {response.status_code} - {response.text}")
                return [], []
                
            data = response.json()
            
            # Parse NVIDIA API response
            # Structure: data[0].text_detections[].text_prediction.{text, confidence}
            if 'data' in data and len(data['data']) > 0:
                detections = data['data'][0].get('text_detections', [])
                for detection in detections:
                    prediction = detection.get('text_prediction', {})
                    text = prediction.get('text', '')
                    conf = prediction.get('confidence', 0.0)
                    
                    if text and conf > OCR_CONFIDENCE_THRESHOLD:
                        # Clean text (remove spaces often found in API response)
                        # API returns "JK01 AF5936", we might want "JK01AF5936" 
                        # but validate_indian_plate handles space removal/cleanup
                        all_texts.append(text)
                        all_confidences.append(float(conf))
                        
        except Exception as e:
            logger.error(f"OCR API execution error: {e}")
        
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
        logger.info("Detecting plate...")
        
        # Capture memory before processing
        mem_before = get_detailed_memory_usage()
        
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
            
            # Get memory stats after processing
            mem_after = get_detailed_memory_usage()
            process_mem_mb = mem_after['process_rss_mb']
            request_delta_mb = mem_after['process_rss_mb'] - mem_before['process_rss_mb']
            available_mb = mem_after['system_available_mb']
            
            logger.info(
                f"TIMING PROFILE | YOLO: {yolo_time_ms:.1f}ms | OCR: {ocr_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms | "
                f"RAM Total: {process_mem_mb:.1f}MB | Request Delta: {request_delta_mb:+.1f}MB | Available: {available_mb:.0f}MB"
            )
            
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