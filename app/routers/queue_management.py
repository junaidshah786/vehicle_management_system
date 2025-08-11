from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from app.config.config import VEHICLE_QUEUE_COLLECTION, CLEAN_INACTIVE_SESSIONS_TIME
from app.services.pydantic import QRRequest, RegisterDeviceRequest
import logging
import traceback
from app.services.firebase import db
from typing import  Dict, Any
from fastapi import Query
from typing import Dict, Any
from app.services.vehicle_queue_utils import add_vehicle_to_queue_firestore, fetch_vehicles_by_type_sorted, get_vehicle_details_firestore, get_vehicle_from_queue, log_queue_history_firestore, release_vehicle_from_queue_firestore, update_queue_ranks_after_removal
router = APIRouter()


@router.get("/queue", response_model=Dict[str, Any])
async def fetch_queue(
    vehicle_type: str = Query(..., description="Type of vehicle to filter by"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    try:
        all_vehicles = fetch_vehicles_by_type_sorted(vehicle_type)
        
        if not all_vehicles:
            return {"message": "No vehicles found in queue", "data": []}

        start = (page - 1) * limit
        end = start + limit
        paginated_vehicles = all_vehicles[start:end]

        return {
            "message": "Vehicles fetched successfully",
            "total": len(all_vehicles),
            "page": page,
            "limit": limit,
            "data": paginated_vehicles
        }

    except Exception as e:
        logging.error(f"Error in fetch_queue: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch vehicle queue")




@router.post("/verify-and-queue-vehicle")
async def verify_and_queue_vehicle(request: QRRequest):
    try:
        if "_" not in request.qr_data:
            raise HTTPException(status_code=400, detail="Invalid QR format")
        
        vehicle_id, registration_number, vehicle_type = request.qr_data.split("__", 2)
        logging.info(f"Processing vehicle ID: {vehicle_id}, Registration: {registration_number}, Type: {vehicle_type}")
        
        # Check if vehicle already in queue (directly from Firestore)
        existing_entry = await get_vehicle_from_queue(vehicle_id)

        if existing_entry:
            current_rank = existing_entry.get("queue_rank")
            vehicle_type = existing_entry.get("vehicle_type")

            # 🚫 Reject if rank is not 1
            if current_rank != 1:
                raise HTTPException(
                    status_code=409,
                    detail="Only vehicle on TOP (rank 1) can be released"
                )

            # ✅ Proceed to release - Direct Firestore DELETE (triggers Cloud Function automatically)
            await release_vehicle_from_queue_firestore(vehicle_id)
            
            # Update ranks of remaining vehicles
            await update_queue_ranks_after_removal(current_rank, vehicle_type)

            # Log history in Firestore
            await log_queue_history_firestore(vehicle_id, "removed", current_rank, vehicle_type, request.name)

            return {
                "message": "Vehicle released from queue",
                "vehicleId": vehicle_id,
                "vehicleType": vehicle_type,
                "previousQueueRank": current_rank
            }

        # Not in queue — fetch vehicle details from Firestore
        vehicle_data = await get_vehicle_details_firestore(vehicle_id)
        if not vehicle_data:
            logging.error(f"Vehicle with ID {vehicle_id} not found")
            raise HTTPException(status_code=404, detail="Vehicle not found")

        if vehicle_data.get("registrationNumber") != registration_number:
            logging.error(f"Registration number mismatch")
            raise HTTPException(status_code=403, detail="Registration number mismatch")

        vehicle_type = vehicle_data.get("vehicleType")
        if not vehicle_type:
            logging.error(f"Vehicle type missing for vehicle ID {vehicle_id}")
            raise HTTPException(status_code=400, detail="Vehicle type missing")

        # Add to queue - Direct Firestore CREATE (triggers Cloud Function automatically)
        next_rank = await add_vehicle_to_queue_firestore(vehicle_id, registration_number, vehicle_type, request.name)

        # Log history in Firestore
        await log_queue_history_firestore(vehicle_id, "added", next_rank, vehicle_type, request.name)

        return {
            "message": "Vehicle added to queue successfully",
            "vehicleId": vehicle_id,
            "queueRank": next_rank,
            "vehicleType": vehicle_type
        }

    except Exception as e:
        logging.error(f"Error in verify_and_add_to_queue: {traceback.format_exc()}")
        raise HTTPException(status_code= e.status_code if hasattr(e, 'status_code') else 500, detail=str(e))

# Direct Firestore operations (NO manual notification calls)


# Additional endpoints that work directly with Firestore

# @router.get("/queue")
# async def get_current_queue(vehicle_type: Optional[str] = None):
#     """Get current queue from Firestore"""
#     try:
#         query = db.collection(VEHICLE_QUEUE_COLLECTION)
        
#         if vehicle_type:
#             query = query.where("vehicle_type", "==", vehicle_type)
            
#         query = query.order_by("queue_rank")
        
#         docs = query.stream()
#         queue = []
        
#         for doc in docs:
#             data = doc.to_dict()
#             queue.append(data)
            
#         return {"queue": queue, "count": len(queue)}
        
#     except Exception as e:
#         logging.error(f"Error getting current queue: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.put("/queue/{vehicle_id}/status")
# async def update_vehicle_status(vehicle_id: str, new_status: str):
#     """Update vehicle status - DIRECT FIRESTORE UPDATE (triggers Cloud Function)"""
#     try:
#         doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        
#         # Check if vehicle exists
#         if not doc_ref.get().exists:
#             raise HTTPException(status_code=404, detail="Vehicle not found in queue")
        
#         # DIRECT FIRESTORE UPDATE - This automatically triggers Cloud Function
#         doc_ref.update({
#             "status": new_status,
#             "updated_at": firestore.SERVER_TIMESTAMP
#         })
        
#         return {"message": f"Vehicle {vehicle_id} status updated to {new_status}"}
        
#     except Exception as e:
#         logging.error(f"Error updating vehicle status: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/queue/reorder")
# async def reorder_queue(new_order: List[Dict[str, Any]]):
#     """Reorder entire queue - DIRECT FIRESTORE BATCH UPDATE (triggers Cloud Function)"""
#     try:
#         batch = db.batch()
        
#         for item in new_order:
#             vehicle_id = item.get("vehicle_id")
#             new_rank = item.get("queue_rank")
            
#             if not vehicle_id or not new_rank:
#                 continue
                
#             doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
            
#             # DIRECT FIRESTORE UPDATE - Each update automatically triggers Cloud Function
#             batch.update(doc_ref, {
#                 "queue_rank": new_rank,
#                 "updated_at": firestore.SERVER_TIMESTAMP
#             })
        
#         batch.commit()
        
#         return {"message": "Queue reordered successfully"}
        
#     except Exception as e:
#         logging.error(f"Error reordering queue: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@router.delete("/queue/{vehicle_id}")
async def remove_vehicle_from_queue(vehicle_id: str):
    """Remove vehicle from queue - DIRECT FIRESTORE DELETE (triggers Cloud Function)"""
    try:
        doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        
        # Get vehicle data before deletion
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Vehicle not found in queue")
        
        vehicle_data = doc.to_dict()
        removed_rank = vehicle_data.get("queue_rank")
        vehicle_type = vehicle_data.get("vehicle_type")
        
        # DIRECT FIRESTORE DELETE - This automatically triggers Cloud Function
        doc_ref.delete()
        
        # Update remaining vehicles' ranks
        await update_queue_ranks_after_removal(removed_rank, vehicle_type)
        
        return {"message": f"Vehicle {vehicle_id} removed from queue"}
        
    except Exception as e:
        logging.error(f"Error removing vehicle from queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# cloud function calling endpoints (register, unregister, heartbeat)

@router.post("/register-device")
async def register_device_session(device_data: RegisterDeviceRequest):
    """Register device for current app session only"""
    try:
        user_id = device_data.userId
        expo_token = device_data.expoPushToken
        session_id = device_data.sessionId
        
        # Use combination of userId and sessionId as document ID
        doc_id = f"{user_id}_{session_id}"
        
        device_doc = {
            "userId": user_id,
            "expoPushToken": expo_token,
            "sessionId": session_id,
            "isActive": True,
            "registeredAt": datetime.utcnow(),
            "lastSeen": datetime.utcnow()
        }
        
        # Store in activeDevices collection (temporary, session-based)
        doc_ref = db.collection("activeDevices").document(doc_id)
        doc_ref.set(device_doc)
        
        return {"message": "Device registered for current session"}
        
    except Exception as e:
        logging.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unregister-device")
async def unregister_device_session(device_data: RegisterDeviceRequest):
    """Unregister device when app goes inactive"""
    try:
        user_id = device_data.userId
        session_id = device_data.sessionId
        
        doc_id = f"{user_id}_{session_id}"
        doc_ref = db.collection("activeDevices").document(doc_id)
        
        # Remove the device from active list
        doc_ref.delete()
        
        return {"message": "Device unregistered"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/heartbeat")
async def device_heartbeat(heartbeat_data: RegisterDeviceRequest ):
    """Keep device registration alive while app is active"""
    try:
        user_id = heartbeat_data.userId
        session_id = heartbeat_data.sessionId
        
        doc_id = f"{user_id}_{session_id}"
        doc_ref = db.collection("activeDevices").document(doc_id)
        
        # Update last seen timestamp
        doc_ref.update({"lastSeen": datetime.utcnow()})
        
        return {"message": "Heartbeat recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Background cleanup function (run periodically)
@router.post("/cleanup-inactive-devices")
async def cleanup_inactive_devices():
    """Remove devices that haven't been seen recently"""
    try:
        cutoff_time = datetime.now() - timedelta(minutes=CLEAN_INACTIVE_SESSIONS_TIME)  # 15 minutes ago
        
        query = db.collection("activeDevices")\
                 .where("lastSeen", "<", cutoff_time)
        
        docs = query.stream()
        batch = db.batch()
        
        for doc in docs:
            batch.delete(doc.reference)
        
        batch.commit()
        
        return {"message": "Inactive devices cleaned up"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))