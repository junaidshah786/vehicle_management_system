# JWT setup
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# Firestore configuration
vehicle_collection_name = "vehicleDetails"
firebase_credentials_path = "./vehiclemanagementsystem-e76c4-firebase-adminsdk-fbsvc-db53f875ef.json"

# Collections
VEHICLE_QUEUE_COLLECTION = "vehicleQueue"
VEHICLES_COLLECTION = "vehicleDetails"
QUEUE_HISTORY_COLLECTION = "vehicleQueueHistory"
ACTIVE_TRIPS_COLLECTION = "activeTrips"

CLEAN_INACTIVE_SESSIONS_TIME = 15  # minutes

# Plate Detection Settings
YOLO_MODEL_PATH = "models/license-plate-finetune-v1s.pt"
DETECTION_CONFIDENCE = 0.5      # YOLO detection confidence threshold
OCR_CONFIDENCE_THRESHOLD = 0.6  # Minimum OCR confidence to accept plate
MIN_PLATE_WIDTH = 60            # Minimum plate width in pixels
MIN_PLATE_HEIGHT = 15           # Minimum plate height in pixels
MAX_IMAGE_SIZE_MB = 10          # Maximum upload image size in MB

# NVIDIA OCR API
NVIDIA_OCR_API_KEY = "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"
NVIDIA_OCR_API_URL = "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"
