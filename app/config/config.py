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
