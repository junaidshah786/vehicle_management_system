"""
Lightweight License Plate Detection for Low-End Systems
Optimized for: 2GB RAM, No GPU, CPU-only processing
Uses: Tesseract OCR (lightweight) + OpenCV
"""

import cv2
import pytesseract
import numpy as np
from pathlib import Path
import re
import os

class LightweightPlateDetector:
    def __init__(self):
        """
        Initialize lightweight detector
        Make sure Tesseract is installed:
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
        Linux: sudo apt-get install tesseract-ocr
        """
        # Set Tesseract path (Windows - adjust if needed)
        # Uncomment and modify if Tesseract is in a different location
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Configure Tesseract for better plate recognition
        self.tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        print("Lightweight detector initialized (CPU-only)")
    
    def enhance_plate_region(self, roi):
        """Enhance image quality for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def detect_plate_regions(self, frame):
        """
        Detect potential license plate regions using contours
        Memory efficient approach
        """
        # Resize frame if too large (save memory)
        height, width = frame.shape[:2]
        max_width = 800
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter (reduce noise, keep edges)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area, keep top 30
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_regions = []
        
        for contour in contours:
            # Approximate contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            # License plates are typically rectangular
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Filter by aspect ratio and size
                if 2.0 <= aspect_ratio <= 5.5 and w > 60 and h > 20:
                    plate_regions.append((x, y, w, h))
        
        return plate_regions, frame
    
    def clean_plate_text(self, text):
        """Clean and validate detected text"""
        if not text:
            return None
        
        # Remove special characters and spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Must have both letters and numbers
        if not (re.search(r'[A-Z]', cleaned) and re.search(r'\d', cleaned)):
            return None
        
        # Length check (typical plates are 5-10 characters)
        if len(cleaned) < 5 or len(cleaned) > 10:
            return None
        
        # Common patterns (adjust for your region)
        patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # Indian: DL01AB1234
            r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',        # Indian: DL01A1234
            r'^[A-Z]{3}\d{3,4}$',                 # Simple format
            r'^[A-Z0-9]{5,10}$'                   # Generic
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned):
                return cleaned
        
        return None
    
    def extract_text_from_region(self, frame, x, y, w, h):
        """Extract text using Tesseract OCR"""
        # Add padding
        padding = 5
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        roi = frame[y1:y2, x1:x2]
        
        # Enhance ROI
        enhanced = self.enhance_plate_region(roi)
        
        # OCR with Tesseract
        try:
            text = pytesseract.image_to_string(enhanced, config=self.tesseract_config)
            return text.strip()
        except Exception as e:
            return None
    
    def process_video(self, video_path, frame_skip=30, max_frames=None):
        """
        Process video file (memory efficient)
        frame_skip: Process every Nth frame (higher = faster, less memory)
        max_frames: Limit total frames processed (None = all frames)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        detected_plates = set()
        frame_count = 0
        processed_count = 0
        
        print(f"\nProcessing: {Path(video_path).name}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Limit processing for very long videos
            if max_frames and processed_count >= max_frames:
                print(f"  Reached max frames limit ({max_frames})")
                break
            
            # Progress
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} frames ({frame_count}/{total_frames} total)...")
            
            # Detect plate regions
            plate_regions, resized_frame = self.detect_plate_regions(frame)
            
            # Extract text from regions
            for (x, y, w, h) in plate_regions:
                text = self.extract_text_from_region(resized_frame, x, y, w, h)
                cleaned = self.clean_plate_text(text)
                
                if cleaned and cleaned not in detected_plates:
                    detected_plates.add(cleaned)
                    print(f"  ✓ Detected: {cleaned}")
            
            # Free memory
            del frame
        
        cap.release()
        return list(detected_plates)
    
    def process_folder(self, folder_path, frame_skip=30, max_frames=200):
        """
        Process all videos in folder
        Optimized for low-memory systems
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: Folder {folder_path} does not exist")
            return
        
        # Video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(folder.glob(f'*{ext}'))
            video_files.extend(folder.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No video files found in {folder_path}")
            return
        
        print(f"Found {len(video_files)} video file(s)")
        print(f"Settings: frame_skip={frame_skip}, max_frames={max_frames}")
        print(f"(Optimized for low-memory systems)\n")
        
        all_plates = {}
        
        for video_file in video_files:
            plates = self.process_video(str(video_file), frame_skip, max_frames)
            all_plates[video_file.name] = plates
        
        return all_plates


def main():
    """Main function"""
    print("="*60)
    print("Lightweight License Plate Detection System")
    print("Optimized for: Low-end systems (2GB RAM, CPU only)")
    print("="*60)
    
    # Initialize detector
    detector = LightweightPlateDetector()
    
    # Folder path
    folder_path = r"E:\workspace\poc\vehicle_management_system\footages"
    
    # Process videos
    # Adjust these parameters based on your system:
    # - frame_skip: Higher = faster but might miss plates (20-60 recommended)
    # - max_frames: Limit frames per video to save memory (100-300 recommended)
    results = detector.process_folder(
        folder_path, 
        frame_skip=30,      # Process every 30th frame
        max_frames=200      # Process max 200 frames per video
    )
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    if results:
        for video_name, plates in results.items():
            print(f"\n{video_name}:")
            if plates:
                for plate in plates:
                    print(f"  • {plate}")
            else:
                print("  No plates detected")
    else:
        print("\nNo results")
    
    print("\n" + "="*60)
    print("TIP: If accuracy is low, try:")
    print("  - Decrease frame_skip (e.g., 20 instead of 30)")
    print("  - Increase max_frames if memory allows")
    print("  - Ensure Tesseract is properly installed")
    print("="*60)


if __name__ == "__main__":
    main()