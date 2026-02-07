import logging
import traceback
from app.config.config import QUEUE_HISTORY_COLLECTION, VEHICLE_QUEUE_COLLECTION, VEHICLES_COLLECTION
from firebase_admin import firestore
from datetime import datetime
from app.services.firebase import db
from typing import List, Dict, Any, Optional



# def release_vehicle_from_queue(doc_id: str, vehicle_type: str, current_rank: int):
#     queue_ref = db.collection("vehicleQueue")

#     # Delete the current entry
#     queue_ref.document(doc_id).delete()

#     # Update lower ranked vehicles
#     lower_ranked_query = queue_ref \
#         .where("vehicle_type", "==", vehicle_type) \
#         .where("queue_rank", ">", current_rank) \
#         .stream()

#     batch = db.batch()
#     for doc in lower_ranked_query:
#         doc_ref = queue_ref.document(doc.id)
#         doc_data = doc.to_dict()
#         updated_rank = doc_data["queue_rank"] - 1
#         batch.update(doc_ref, {"queue_rank": updated_rank})
#     batch.commit()


# def fetch_vehicle_details(vehicle_id: str):
#     doc = db.collection("vehicleDetails").document(vehicle_id).get()
#     if not doc.exists:
#         return None
#     return doc.to_dict()


def get_next_queue_rank(vehicle_type: str) -> int:
    query = db.collection("vehicleQueue") \
              .where("vehicle_type", "==", vehicle_type) \
              .order_by("queue_rank", direction=firestore.Query.DESCENDING) \
              .limit(1).stream()

    for doc in query:
        return doc.to_dict().get("queue_rank", 0) + 1
    return 1


# def add_vehicle_to_queue(vehicle_id: str, registration_number: str, vehicle_type: str, username: str) -> int:
#     next_rank = get_next_queue_rank(vehicle_type)

#     queue_entry = {
#         "vehicle_id": vehicle_id,
#         "registration_number": registration_number,
#         "vehicle_type": vehicle_type,
#         "username": username,
#         "queue_rank": next_rank,
#         "status": "waiting",
#         "added_at": datetime.utcnow()
#     }

#     db.collection("vehicleQueue").add(queue_entry)
#     return next_rank


# def log_queue_history(vehicle_id: str, action: str, queue_rank: int, vehicle_type: str, username: str):
#     history_entry = {
#         "vehicle_id": vehicle_id,
#         "action": action,  # "added" or "removed"
#         "queue_rank": queue_rank,
#         "vehicle_type": vehicle_type,
#         "username": username,
#         "timestamp": datetime.utcnow()
#     }
#     db.collection("vehicleQueueHistory").add(history_entry)



# def fetch_vehicles_by_type_sorted(vehicle_type: str) -> List[Dict[str, Any]]:
#     """
#     Fetches all vehicles of a specific type from the queue,
#     sorted by their queue_rank (ascending).
#     """
#     try:
#         docs = db.collection("vehicleQueue").where("vehicle_type", "==", vehicle_type).stream()

#         vehicles = []
#         for doc in docs:
#             data = doc.to_dict()
#             # Ensure vehicleShift is included if present
#             if "vehicleShift" not in data:
#                 # Try to fetch from vehicle details if missing
#                 vehicle_id = data.get("vehicle_id")
#                 if vehicle_id:
#                     vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
#                     if vehicle_details.exists:
#                         data["vehicleShift"] = vehicle_details.to_dict().get("vehicleShift")
#             vehicles.append(data)

#         vehicles.sort(key=lambda x: x.get("queue_rank", float("inf")))
#         return vehicles
#     except Exception as e:
#         logging.error(f"Error fetching vehicles by type {vehicle_type}: {e}")
#         return []


def fetch_vehicles_by_type_sorted(vehicle_type: str) -> List[Dict[str, Any]]:
    """
    Fetches all vehicles of a specific type from the queue,
    sorted by their queue_rank (ascending).
    """
    try:
        docs = db.collection("vehicleQueue").where("vehicle_type", "==", vehicle_type).stream()

        vehicles = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure vehicleShift is included if present
            if "vehicleShift" not in data:
                # Try to fetch from vehicle details if missing
                vehicle_id = data.get("vehicle_id")
                if vehicle_id:
                    vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
                    if vehicle_details.exists:
                        data["vehicleShift"] = vehicle_details.to_dict().get("vehicleShift")
            vehicles.append(data)

        vehicles.sort(key=lambda x: x.get("queue_rank", float("inf")))
        return vehicles
    except Exception as e:
        logging.error(f"Error fetching vehicles by type {vehicle_type}: {e}")
        return []
    
    
def fetch_all_vehicles_sorted() -> List[Dict[str, Any]]:
    """
    Fetches all vehicles from the queue,
    ensuring vehicleShift is included if present,
    sorted by queue_rank (ascending).
    """
    try:
        docs = db.collection("vehicleQueue").stream()

        vehicles = []
        for doc in docs:
            data = doc.to_dict()
            
            # Ensure vehicleShift is included if present
            if "vehicleShift" not in data:
                vehicle_id = data.get("vehicle_id")
                if vehicle_id:
                    vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
                    if vehicle_details.exists:
                        data["vehicleShift"] = vehicle_details.to_dict().get("vehicleShift")
            
            vehicles.append(data)

        # Sort vehicles by queue_rank
        vehicles.sort(key=lambda x: x.get("queue_rank", float("inf")))
        return vehicles

    except Exception as e:
        logging.error(f"Error fetching all vehicles: {e}")
        return []



async def get_vehicle_from_queue(vehicle_id: str) -> Optional[Dict]:
    """Get vehicle from Firestore queue"""
    try:
        doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logging.error(f"Error getting vehicle from queue: {e}")
        return None

async def get_vehicle_details_firestore(vehicle_id: str) -> Optional[Dict]:
    """Get vehicle details from Firestore vehicles collection"""
    try:
        doc_ref = db.collection(VEHICLES_COLLECTION).document(vehicle_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logging.error(f"Error getting vehicle details: {e}")
        return None

async def add_vehicle_to_queue_firestore(vehicle_id: str, registration_number: str, vehicle_type: str, driver_name: str, vehicle_shift:str) -> int:
    
    """Add vehicle to Firestore queue - DIRECT WRITE (triggers Cloud Function)"""
    try:
        # Get next rank
        next_rank = await get_next_queue_rank(vehicle_type, vehicle_shift)
        
        # Fetch vehicleShift from vehicle details
        vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
        if vehicle_details.exists:
            vehicle_shift = vehicle_details.to_dict().get("vehicleShift")
        
        # Create vehicle queue entry
        vehicle_queue_data = {
            "vehicle_id": vehicle_id,
            "registration_number": registration_number,
            "vehicle_type": vehicle_type,
            "queue_rank": next_rank,
            "driver_name": driver_name,
            "status": "waiting",
            "queued_at": datetime.utcnow(),
            "created_by": driver_name,
            "vehicleShift": vehicle_shift  # <-- Added
        }
        
        # DIRECT FIRESTORE WRITE - This automatically triggers Cloud Function
        doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        doc_ref.set(vehicle_queue_data)
        
        logging.info(f"Vehicle {vehicle_id} added to Firestore queue at rank {next_rank}")
        return next_rank
        
    except Exception as e:
        logging.error(f"Error adding vehicle to queue: {e}")
        raise

async def release_vehicle_from_queue_firestore(vehicle_id: str):
    """Remove vehicle from Firestore queue - DIRECT DELETE (triggers Cloud Function)"""
    try:
        # DIRECT FIRESTORE DELETE - This automatically triggers Cloud Function
        doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        doc_ref.delete()
        
        logging.info(f"Vehicle {vehicle_id} released from Firestore queue")
        
    except Exception as e:
        logging.error(f"Error releasing vehicle from queue: {e}")
        raise


async def get_next_queue_rank(vehicle_type: str, vehicle_shift: str) -> int:
    """Get next rank for vehicle type and shift"""
    try:
        # Query vehicles of same type AND same shift
        query = db.collection(VEHICLE_QUEUE_COLLECTION)\
                 .where("vehicle_type", "==", vehicle_type)\
                 .where("vehicleShift", "==", vehicle_shift)\
                 .order_by("queue_rank", direction=firestore.Query.DESCENDING)\
                 .limit(1)
        
        docs = query.stream()
        max_rank = 0
        
        for doc in docs:
            max_rank = doc.to_dict().get("queue_rank", 0)
            break
            
        return max_rank + 1
        
    except Exception as e:
        logging.error(f"Error getting next queue rank: {e}")
        raise Exception("Failed to get next queue rank")


async def update_queue_ranks_after_removal(removed_rank: int, vehicle_type: str):
    """Update ranks of vehicles after one is removed - DIRECT FIRESTORE UPDATES (triggers Cloud Function)"""
    try:
        # Get all vehicles with higher ranks of same type
        query = db.collection(VEHICLE_QUEUE_COLLECTION)\
               .where("vehicle_type", "==", vehicle_type)\
               .where("queue_rank", ">", removed_rank)
        
        docs = query.stream()
        batch = db.batch()
        
        for doc in docs:
            current_rank = doc.to_dict().get("queue_rank")
            new_rank = current_rank - 1
            
            # DIRECT FIRESTORE UPDATE - This automatically triggers Cloud Function
            batch.update(doc.reference, {"queue_rank": new_rank, "updated_at": datetime.utcnow()})
        
        batch.commit()
        logging.info(f"Updated ranks after removing vehicle at rank {removed_rank}")
        
    except Exception as e:
        logging.error(f"Error updating queue ranks: {e}")


# async def log_queue_history_firestore(vehicle_id: str, action: str, rank: int, vehicle_type: str, user: str, checkin_time: datetime = None, checkout_time: datetime = None):
#     """Log queue history to Firestore with check-in and check-out times"""
#     try:
#         # Fetch vehicleShift from vehicle details
#         vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
#         vehicle_shift = None
#         if vehicle_details.exists:
#             vehicle_shift = vehicle_details.to_dict().get("vehicleShift")
#         history_data = {
#             "vehicle_id": vehicle_id,
#             "action": action,
#             "queue_rank": rank,
#             "vehicle_type": vehicle_type,
#             "vehicleShift": vehicle_shift,  # <-- Added
#             "username": user,
#             "timestamp": datetime.utcnow(),
#             "checkin_time": checkin_time,
#             "checkout_time": checkout_time
#         }
#         db.collection(QUEUE_HISTORY_COLLECTION).add(history_data)
#     except Exception as e:
#         logging.error(f"Error logging queue history: {e}")


async def log_queue_history_firestore(
    vehicle_id: str, 
    action: str, 
    rank: int, 
    vehicle_type: str, 
    user: str, 
    checkin_time: datetime = None, 
    checkout_time: datetime = None,
    released_time: datetime = None
):
    """
    Log queue history to Firestore with complete lifecycle tracking
    
    Args:
        vehicle_id: Unique identifier for the vehicle
        action: Action type - 'checked_in', 'released', 'checked_out', 'added', 'removed'
        rank: Queue rank at the time of action
        vehicle_type: Type of vehicle
        user: Username performing the action
        checkin_time: When vehicle was added to queue
        checkout_time: When vehicle completed the entire cycle
        released_time: When vehicle was released from queue for trip
    """
    try:
        # Fetch vehicleShift from vehicle details
        vehicle_details = db.collection("vehicleDetails").document(vehicle_id).get()
        vehicle_shift = None
        registration_number = None
        
        if vehicle_details.exists:
            vehicle_data = vehicle_details.to_dict()
            vehicle_shift = vehicle_data.get("vehicleShift")
            registration_number = vehicle_data.get("registrationNumber")
        
        # Helper function to normalize datetime (handle DatetimeWithNanoseconds)
        def normalize_datetime(dt):
            """Convert to standard Python datetime to avoid Firestore issues"""
            if dt is None:
                return None
            try:
                # Create fresh datetime to avoid DatetimeWithNanoseconds issues
                return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, getattr(dt, 'microsecond', 0))
            except Exception:
                if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                    return dt.replace(tzinfo=None)
                return dt
        
        # Normalize all datetime objects
        checkin_time = normalize_datetime(checkin_time)
        released_time = normalize_datetime(released_time)
        checkout_time = normalize_datetime(checkout_time)
        
        # Calculate durations if applicable
        queue_duration = None
        trip_duration = None
        total_duration = None
        
        if checkin_time and released_time:
            queue_duration = (released_time - checkin_time).total_seconds()
        
        if released_time and checkout_time:
            trip_duration = (checkout_time - released_time).total_seconds()
        
        if checkin_time and checkout_time:
            total_duration = (checkout_time - checkin_time).total_seconds()
        
        history_data = {
            "vehicle_id": vehicle_id,
            "registration_number": registration_number,
            "action": action,
            "queue_rank": rank,
            "vehicle_type": vehicle_type,
            "vehicleShift": vehicle_shift,
            "username": user,
            "timestamp": datetime.utcnow(),
            
            # Lifecycle timestamps
            "checkin_time": checkin_time,
            "released_time": released_time,
            "checkout_time": checkout_time,
            
            # Durations (in seconds)
            "queue_duration_seconds": queue_duration,
            "trip_duration_seconds": trip_duration,
            "total_duration_seconds": total_duration,
            
            # Status indicator
            "status": get_status_from_action(action)
        }
        
        # Add to history collection
        db.collection(QUEUE_HISTORY_COLLECTION).add(history_data)
        
        logging.info(f"Queue history logged: {vehicle_id} - {action}")
        
    except Exception as e:
        logging.error(f"Error logging queue history for {vehicle_id}: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")


def get_status_from_action(action: str) -> str:
    """Map action to status for easier filtering"""
    status_map = {
        "checked_in": "in_queue",
        "added": "in_queue",
        "released": "on_trip",
        "checked_out": "completed",
        "removed": "removed"
    }
    return status_map.get(action, "unknown")


async def get_vehicle_history(vehicle_id: str) -> List[Dict]:
    """
    Fetch complete history for a specific vehicle
    
    Returns list of history entries sorted by timestamp
    """
    try:
        history_ref = db.collection(QUEUE_HISTORY_COLLECTION)
        query = history_ref.where("vehicle_id", "==", vehicle_id).order_by("timestamp", direction="DESCENDING")
        
        docs = query.stream()
        history = []
        
        for doc in docs:
            entry = doc.to_dict()
            entry["id"] = doc.id
            history.append(entry)
        
        return history
        
    except Exception as e:
        logging.error(f"Error fetching vehicle history: {e}")
        return []


async def get_trip_statistics(
    start_date: datetime = None, 
    end_date: datetime = None,
    vehicle_type: str = None,
    vehicle_shift: str = None
) -> Dict:
    """
    Calculate trip statistics for reporting
    
    Args:
        start_date: Filter from this date
        end_date: Filter until this date
        vehicle_type: Filter by vehicle type
        vehicle_shift: Filter by vehicle shift
    
    Returns:
        Dictionary with statistics
    """
    try:
        history_ref = db.collection(QUEUE_HISTORY_COLLECTION)
        query = history_ref.where("action", "==", "checked_out")
        
        if start_date:
            query = query.where("checkin_time", ">=", start_date)
        if end_date:
            query = query.where("checkin_time", "<=", end_date)
        if vehicle_type:
            query = query.where("vehicle_type", "==", vehicle_type)
        if vehicle_shift:
            query = query.where("vehicleShift", "==", vehicle_shift)
        
        docs = query.stream()
        
        total_trips = 0
        total_queue_time = 0
        total_trip_time = 0
        total_overall_time = 0
        
        for doc in docs:
            data = doc.to_dict()
            total_trips += 1
            
            if data.get("queue_duration_seconds"):
                total_queue_time += data["queue_duration_seconds"]
            if data.get("trip_duration_seconds"):
                total_trip_time += data["trip_duration_seconds"]
            if data.get("total_duration_seconds"):
                total_overall_time += data["total_duration_seconds"]
        
        return {
            "total_trips": total_trips,
            "average_queue_time_minutes": (total_queue_time / total_trips / 60) if total_trips > 0 else 0,
            "average_trip_time_minutes": (total_trip_time / total_trips / 60) if total_trips > 0 else 0,
            "average_total_time_minutes": (total_overall_time / total_trips / 60) if total_trips > 0 else 0,
            "total_queue_time_hours": total_queue_time / 3600,
            "total_trip_time_hours": total_trip_time / 3600,
            "filters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "vehicle_type": vehicle_type,
                "vehicle_shift": vehicle_shift
            }
        }
        
    except Exception as e:
        logging.error(f"Error calculating trip statistics: {e}")
        return {"error": str(e)}
    

async def cleanup_invalid_tokens(invalid_tokens: List[str]):
    """Remove invalid FCM tokens from activeDevices collection"""
    try:
        if not invalid_tokens:
            return
            
        # Query and remove devices with invalid tokens
        active_devices_ref = db.collection("activeDevices")
        
        for token in invalid_tokens:
            # Find and delete devices with this token
            query = active_devices_ref.where("fcm_token", "==", token)
            docs = query.stream()
            
            for doc in docs:
                doc.reference.delete()
                logging.info(f"Removed invalid token: {token}")
                
    except Exception as e:
        logging.error(f"Error cleaning up invalid tokens: {e}")

