import logging
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
            # data["doc_id"] = doc.id
            # data.pop("vehicle", None)
            vehicles.append(data)

        vehicles.sort(key=lambda x: x.get("queue_rank", float("inf")))
        return vehicles
    except Exception as e:
        logging.error(f"Error fetching vehicles by type {vehicle_type}: {e}")
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

async def add_vehicle_to_queue_firestore(vehicle_id: str, registration_number: str, vehicle_type: str, driver_name: str) -> int:
    """Add vehicle to Firestore queue - DIRECT WRITE (triggers Cloud Function)"""
    try:
        # Get next rank
        next_rank = await get_next_queue_rank(vehicle_type)
        
        # Create vehicle queue entry
        vehicle_queue_data = {
            "vehicle_id": vehicle_id,
            "registration_number": registration_number,
            "vehicle_type": vehicle_type,
            "queue_rank": next_rank,
            "driver_name": driver_name,
            "status": "waiting",
            "queued_at": datetime.utcnow(),
            "created_by": driver_name
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


async def get_next_queue_rank(vehicle_type: str) -> int:
    """Get next rank for vehicle type"""
    try:
        # Query vehicles of same type to get max rank
        query = db.collection(VEHICLE_QUEUE_COLLECTION)\
                 .where("vehicle_type", "==", vehicle_type)\
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
        return 1

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

# async def log_queue_history_firestore(vehicle_id: str, action: str, rank: int, vehicle_type: str, user: str):
#     """Log queue history to Firestore"""
#     try:
#         history_data = {
#             "vehicle_id": vehicle_id,
#             "action": action,
#             "queue_rank": rank,
#             "vehicle_type": vehicle_type,
#             "username": user,
#             "timestamp": datetime.utcnow()
#         }
        
#         db.collection(QUEUE_HISTORY_COLLECTION).add(history_data)
        
#     except Exception as e:
#         logging.error(f"Error logging queue history: {e}")



async def log_queue_history_firestore(vehicle_id: str, action: str, rank: int, vehicle_type: str, user: str, checkin_time: datetime = None, checkout_time: datetime = None):
    """Log queue history to Firestore with check-in and check-out times"""
    try:
        history_data = {
            "vehicle_id": vehicle_id,
            "action": action,
            "queue_rank": rank,
            "vehicle_type": vehicle_type,
            "username": user,
            "timestamp": datetime.utcnow(),
            "checkin_time": checkin_time,
            "checkout_time": checkout_time
        }
        db.collection(QUEUE_HISTORY_COLLECTION).add(history_data)
    except Exception as e:
        logging.error(f"Error logging queue history: {e}")


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

