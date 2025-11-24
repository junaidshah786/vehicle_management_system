from collections import defaultdict
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.config.config import VEHICLE_QUEUE_COLLECTION, ACTIVE_TRIPS_COLLECTION
from app.services.pydantic import RegisterDeviceRequest, TestFcmNotificationRequest, UnregisterDeviceRequest
import traceback
from app.services.firebase import db
from fastapi import Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any
from firebase_admin import messaging
import asyncio
from pydantic import BaseModel
import logging
from app.services.vehicle_queue_utils import (
    add_vehicle_to_queue_firestore, 
    cleanup_invalid_tokens, 
    fetch_all_vehicles_sorted, 
    fetch_vehicles_by_type_sorted,
    get_trip_statistics, 
    get_vehicle_details_firestore, 
    get_vehicle_from_queue,
    get_vehicle_history, 
    log_queue_history_firestore, 
    release_vehicle_from_queue_firestore, 
    update_queue_ranks_after_removal
)

router = APIRouter()
logger = logging.getLogger(__name__)

subscribers = []  # Active SSE connections

# ============ REQUEST MODELS ============
class CheckInRequest(BaseModel):
    qr_data: str
    name: str
    contact: str

class ReleaseRequest(BaseModel):
    qr_data: str
    name: str

class CheckOutRequest(BaseModel):
    qr_data: str
    name: str

# ============ HELPER FUNCTIONS ============
async def get_active_trip(vehicle_id: str) -> Dict:
    """Get vehicle from active trips collection"""
    try:
        doc = db.collection(ACTIVE_TRIPS_COLLECTION).document(vehicle_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logging.error(f"Error fetching active trip: {e}")
        return None

async def move_to_active_trips(vehicle_id: str, queue_data: Dict, released_by: str):
    """Move vehicle from queue to active trips"""
    try:
        trip_data = {
            "vehicleId": vehicle_id,
            "registrationNumber": queue_data.get("registration_number"),
            "vehicleType": queue_data.get("vehicle_type"),
            "vehicleShift": queue_data.get("vehicleShift"),
            "driverName": queue_data.get("driver_name"),
            "driverContact": queue_data.get("driver_contact"),
            "queueRank": queue_data.get("queue_rank"),
            "queuedAt": queue_data.get("queued_at"),
            "releasedAt": datetime.utcnow(),
            "releasedBy": released_by,
            "status": "on_trip"
        }
        
        db.collection(ACTIVE_TRIPS_COLLECTION).document(vehicle_id).set(trip_data)
        logging.info(f"Vehicle {vehicle_id} moved to active trips")
        
    except Exception as e:
        logging.error(f"Error moving to active trips: {e}")
        raise

async def remove_from_active_trips(vehicle_id: str):
    """Remove vehicle from active trips collection"""
    try:
        db.collection(ACTIVE_TRIPS_COLLECTION).document(vehicle_id).delete()
        logging.info(f"Vehicle {vehicle_id} removed from active trips")
    except Exception as e:
        logging.error(f"Error removing from active trips: {e}")
        raise

# ============ SSE STREAM ============
@router.get("/notifications/stream")
async def notification_stream(request: Request):
    """Stream real-time notifications to clients."""
    async def event_generator():
        queue = asyncio.Queue()
        subscribers.append(queue)
        try:
            while True:
                if await request.is_disconnected():
                    break
                message = await queue.get()
                yield f"data: {message}\n\n"
        finally:
            subscribers.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

async def broadcast_notification(message: str):
    """Send message to all active SSE subscribers"""
    for queue in subscribers:
        await queue.put(message)

# ============ FETCH ENDPOINTS ============
@router.get("/fetch_queue", response_model=Dict[str, Any])
async def fetch_queue(
    vehicle_type: str = Query(..., description="Type of vehicle to filter by"),
    vehicle_shift: str = Query(None, description="Vehicle shift (morning/day/night)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    try:
        all_vehicles = fetch_vehicles_by_type_sorted(vehicle_type)
        
        if vehicle_shift:
            all_vehicles = [v for v in all_vehicles if v.get("vehicleShift") == vehicle_shift]
        
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

@router.get("/queue", response_model=Dict[str, Any])
async def fetch_all_queue(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    """Fetches all vehicles in the queue, grouped by (vehicle_type + vehicleShift)."""
    try:
        all_vehicles = fetch_all_vehicles_sorted()

        if not all_vehicles:
            return {"message": "No vehicles found in queue", "data": []}

        grouped = defaultdict(list)
        for v in all_vehicles:
            v_type = v.get("vehicle_type", "unknown")
            v_shift = v.get("vehicleShift", "unknown")
            key = f"{v_type}_{v_shift}"
            grouped[key].append(v)

        grouped = {k: v for k, v in grouped.items() if v}
        grouped_list = [{"group": k, "vehicles": v} for k, v in grouped.items()]

        start = (page - 1) * limit
        end = start + limit
        paginated_groups = grouped_list[start:end]

        return {
            "message": "Vehicles fetched successfully",
            "total_groups": len(grouped_list),
            "page": page,
            "limit": limit,
            "data": paginated_groups
        }

    except Exception as e:
        logging.error(f"Error in fetch_queue: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch vehicle queue")

@router.get("/active-trips", response_model=Dict[str, Any])
async def fetch_active_trips(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    """Fetch all vehicles currently on trips"""
    try:
        trips_ref = db.collection(ACTIVE_TRIPS_COLLECTION)
        docs = trips_ref.stream()
        
        all_trips = []
        for doc in docs:
            trip_data = doc.to_dict()
            trip_data["id"] = doc.id
            all_trips.append(trip_data)
        
        # Sort by release time
        all_trips.sort(key=lambda x: x.get("releasedAt", datetime.min), reverse=True)
        
        start = (page - 1) * limit
        end = start + limit
        paginated_trips = all_trips[start:end]
        
        return {
            "message": "Active trips fetched successfully",
            "total": len(all_trips),
            "page": page,
            "limit": limit,
            "data": paginated_trips
        }
        
    except Exception as e:
        logging.error(f"Error fetching active trips: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch active trips")

# ============ FCM FUNCTIONS ============
async def get_active_device_tokens() -> List[str]:
    """Fetch all active device FCM tokens from Firestore"""
    try:
        active_devices_ref = db.collection("activeDevices")
        docs = active_devices_ref.stream()
        
        tokens = []
        for doc in docs:
            device_data = doc.to_dict()
            token = device_data.get("fcmToken")
            is_active = device_data.get("isActive", False)
            
            if token and is_active:
                tokens.append(token)
        
        logging.info(f"Retrieved {len(tokens)} active device tokens")
        return tokens
        
    except Exception as e:
        logging.error(f"Error fetching active device tokens: {e}")
        return []

async def send_fcm_notification_to_active_devices(title: str, body: str, data: Dict = None):
    """Send FCM notification to all active devices"""
    try:
        tokens = await get_active_device_tokens()
        
        if not tokens:
            logging.warning("No active device tokens found")
            return {"total_tokens": 0, "successful": 0, "failed": 0}
        
        notification_data = data or {}
        notification_data.update({
            "action": "refresh_queue",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        string_data = {k: str(v) for k, v in notification_data.items()}
        
        successful_sends = 0
        failed_tokens = []
        
        for i, token in enumerate(tokens):
            try:
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body
                    ),
                    data=string_data,
                    token=token
                )
                
                def send_single_message():
                    return messaging.send(message)
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, send_single_message
                )
                
                successful_sends += 1
                logging.info(f"Successfully sent FCM to token {i+1}/{len(tokens)}")
                
            except Exception as token_error:
                failed_tokens.append(token)
                logging.warning(f"Failed to send to token: {str(token_error)}")
        
        if failed_tokens:
            await cleanup_invalid_tokens(failed_tokens)
        
        return {
            "total_tokens": len(tokens),
            "successful": successful_sends,
            "failed": len(failed_tokens)
        }
        
    except Exception as e:
        logging.error(f"Error in send_fcm_notification_to_active_devices: {e}")
        return {"total_tokens": 0, "successful": 0, "failed": 0, "error": str(e)}

# ============ CHECK-IN API ============
@router.post("/vehicle/check-in")
async def check_in_vehicle(request: CheckInRequest):
    """Add a vehicle to the queue"""
    try:
        if "_" not in request.qr_data:
            raise HTTPException(status_code=400, detail="Invalid QR format")
        
        vehicle_id, registration_number, vehicle_type = request.qr_data.split("__", 2)
        logging.info(f"Check-in: Vehicle ID: {vehicle_id}")
        
        # Check if vehicle already in queue
        existing_entry = await get_vehicle_from_queue(vehicle_id)
        if existing_entry:
            raise HTTPException(
                status_code=409,
                detail=f"Vehicle already in queue at position {existing_entry.get('queue_rank')}"
            )
        
        # Check if vehicle is on an active trip
        active_trip = await get_active_trip(vehicle_id)
        if active_trip:
            raise HTTPException(
                status_code=409,
                detail="Vehicle is currently on a trip. Please check-out first."
            )

        # Fetch vehicle details
        vehicle_data = await get_vehicle_details_firestore(vehicle_id)
        logging.info(f"Fetched vehicle data: {vehicle_data}")
        if not vehicle_data:
            raise HTTPException(status_code=404, detail="Vehicle not found")

        if vehicle_data.get("registrationNumber") != registration_number:
            raise HTTPException(status_code=403, detail="Registration number mismatch")

        vehicle_type = vehicle_data.get("vehicleType")
        vehicle_shift = vehicle_data.get("vehicleShift")
        
        if not vehicle_type:
            raise HTTPException(status_code=400, detail="Vehicle type missing")

        # Add to queue
        next_rank = await add_vehicle_to_queue_firestore(
            vehicle_id, 
            registration_number, 
            vehicle_type, 
            request.name, 
            vehicle_shift
        )

        # Log history
        await log_queue_history_firestore(
            vehicle_id, 
            "checked_in", 
            next_rank, 
            vehicle_type, 
            request.name,
            checkin_time=datetime.utcnow(),
            checkout_time=None
        )

        # Send FCM notification
        # await send_fcm_notification_to_active_devices(
        #     title="Vehicle Checked In",
        #     body=f"{vehicle_type} ({registration_number}) checked in at position {next_rank}",
        #     data={
        #         "vehicle_id": vehicle_id,
        #         "vehicle_type": vehicle_type,
        #         "vehicle_shift": vehicle_shift,
        #         "registration_number": registration_number,
        #         "action": "vehicle_checked_in",
        #         "queue_rank": str(next_rank)
        #     }
        # )

        await broadcast_notification({
            "action": "check_in",
            "vehicle": {
                "vehicleId": vehicle_id,
                "queueRank": next_rank,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "registrationNumber": registration_number
            }
        })

        return {
            "message": "Vehicle checked in successfully",
            "vehicleId": vehicle_id,
            "queueRank": next_rank,
            "vehicleType": vehicle_type,
            "vehicleShift": vehicle_shift,
            "registrationNumber": registration_number
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in check_in_vehicle: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to check in vehicle: {str(e)}")

# ============ RELEASE API (NEW) ============
@router.post("/vehicle/release")
async def release_vehicle(request: ReleaseRequest):
    """Release vehicle from queue when turn arrives (vehicle goes for trip)"""
    try:
        if "_" not in request.qr_data:
            raise HTTPException(status_code=400, detail="Invalid QR format")
        
        vehicle_id, registration_number, vehicle_type = request.qr_data.split("__", 2)
        logging.info(f"Release: Vehicle ID: {vehicle_id}")
        
        # Check if vehicle is in queue
        existing_entry = await get_vehicle_from_queue(vehicle_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="Vehicle not found in queue")

        current_rank = existing_entry.get("queue_rank")
        vehicle_type = existing_entry.get("vehicle_type")
        vehicle_shift = existing_entry.get("vehicleShift")
        
        # Only allow release if rank is 1
        if current_rank != 1:
            raise HTTPException(
                status_code=409,
                detail=f"Only vehicle at position 1 can be released. Current position: {current_rank}"
            )

        # Move to active trips BEFORE removing from queue
        await move_to_active_trips(vehicle_id, existing_entry, request.name)
        
        # Remove from queue
        await release_vehicle_from_queue_firestore(vehicle_id)
        
        # Update ranks of remaining vehicles
        await update_queue_ranks_after_removal(current_rank, vehicle_type)

        # Log history
        await log_queue_history_firestore(
            vehicle_id, 
            "released", 
            current_rank, 
            vehicle_type, 
            request.name,
            checkin_time=existing_entry.get("queued_at"),
            checkout_time=None,
            released_time=datetime.utcnow()
        )

        # Send FCM notification
        # await send_fcm_notification_to_active_devices(
        #     title="Vehicle Released for Trip",
        #     body=f"{vehicle_type} ({registration_number}) released for trip",
        #     data={
        #         "vehicle_id": vehicle_id,
        #         "vehicle_type": vehicle_type,
        #         "vehicle_shift": vehicle_shift,
        #         "registration_number": registration_number,
        #         "action": "vehicle_released",
        #         "previous_rank": str(current_rank)
        #     }
        # )

        await broadcast_notification({
            "action": "release",
            "vehicle": {
                "vehicleId": vehicle_id,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "registrationNumber": registration_number,
                "previousQueueRank": current_rank
            }
        })

        return {
            "message": "Vehicle released for trip",
            "vehicleId": vehicle_id,
            "vehicleType": vehicle_type,
            "vehicleShift": vehicle_shift,
            "registrationNumber": registration_number,
            "previousQueueRank": current_rank,
            "status": "on_trip"
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in release_vehicle: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to release vehicle: {str(e)}")

# ============ CHECK-OUT API ============
@router.post("/vehicle/check-out")
async def check_out_vehicle(request: CheckOutRequest):
    """Check out vehicle after returning from trip"""
    try:
        if "_" not in request.qr_data:
            raise HTTPException(status_code=400, detail="Invalid QR format")
        
        vehicle_id, registration_number, vehicle_type = request.qr_data.split("__", 2)
        logging.info(f"Check-out: Vehicle ID: {vehicle_id}")
        
        # Check if vehicle is on an active trip
        active_trip = await get_active_trip(vehicle_id)
        if not active_trip:
            raise HTTPException(
                status_code=404,
                detail="Vehicle not found in active trips. Cannot check out."
            )

        vehicle_type = active_trip.get("vehicleType")
        vehicle_shift = active_trip.get("vehicleShift")
        queued_at = active_trip.get("queuedAt")
        released_at = active_trip.get("releasedAt")
        
        # Remove from active trips
        await remove_from_active_trips(vehicle_id)

        # Log final history
        await log_queue_history_firestore(
            vehicle_id, 
            "checked_out", 
            active_trip.get("queueRank"),
            vehicle_type, 
            request.name,
            checkin_time=queued_at,
            checkout_time=datetime.utcnow(),
            released_time=released_at
        )

        # Send FCM notification
        # await send_fcm_notification_to_active_devices(
        #     title="Vehicle Checked Out",
        #     body=f"{vehicle_type} ({registration_number}) has completed trip and checked out",
        #     data={
        #         "vehicle_id": vehicle_id,
        #         "vehicle_type": vehicle_type,
        #         "vehicle_shift": vehicle_shift,
        #         "registration_number": registration_number,
        #         "action": "vehicle_checked_out"
        #     }
        # )

        await broadcast_notification({
            "action": "check_out",
            "vehicle": {
                "vehicleId": vehicle_id,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "registrationNumber": registration_number
            }
        })

        return {
            "message": "Vehicle checked out successfully",
            "vehicleId": vehicle_id,
            "vehicleType": vehicle_type,
            "vehicleShift": vehicle_shift,
            "registrationNumber": registration_number,
            "tripDuration": str(datetime.utcnow() - (released_at.replace(tzinfo=None) if released_at and hasattr(released_at, 'tzinfo') and released_at.tzinfo else released_at)) if released_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in check_out_vehicle: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to check out vehicle: {str(e)}")

# ============ DEVICE MANAGEMENT ============
@router.post("/register-device")
async def register_device(device_data: RegisterDeviceRequest):
    """Register a device for FCM notifications"""
    try:
        device_doc_data = {
            "deviceId": device_data.device_id,
            "fcmToken": device_data.fcm_token,
            "username": device_data.username,
            "isActive": True,
            "lastSeen": datetime.utcnow(),
            "registeredAt": datetime.utcnow()
        }
        
        db.collection("activeDevices").document(device_data.device_id).set(
            device_doc_data, merge=True
        )
        
        return {"message": "Device registered successfully"}
        
    except Exception as e:
        logging.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unregister-device")
async def unregister_device(device_data: UnregisterDeviceRequest):
    """Unregister a device from FCM notifications"""
    try:
        db.collection("activeDevices").document(device_data.device_id).update({
            "isActive": False,
            "lastSeen": datetime.utcnow()
        })
        
        return {"message": "Device unregistered successfully"}
        
    except Exception as e:
        logging.error(f"Error unregistering device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-fcm")
async def test_fcm_notification(test_data: TestFcmNotificationRequest):
    """Test FCM notification functionality"""
    try:
        result = await send_fcm_notification_to_active_devices(
            title=test_data.title,
            body=test_data.body,
            data={
                "test": "true",
                "action": "test_notification",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "message": "Test notification process completed",
            "result": result,
            "status": "success" if result.get("successful", 0) > 0 else "no_success"
        }
        
    except Exception as e:
        logging.error(f"Error in test FCM endpoint: {traceback.format_exc()}")
        return {
            "message": "Test notification failed",
            "error": str(e),
            "status": "error"
        }

# ============ ADMIN ENDPOINTS ============
@router.delete("/queue/{vehicle_id}")
async def remove_vehicle_from_queue(vehicle_id: str):
    """Admin: Remove vehicle from queue"""
    try:
        doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
        
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Vehicle not found in queue")
        
        vehicle_data = doc.to_dict()
        removed_rank = vehicle_data.get("queue_rank")
        vehicle_type = vehicle_data.get("vehicle_type")
        
        doc_ref.delete()
        await update_queue_ranks_after_removal(removed_rank, vehicle_type)
        
        return {"message": f"Vehicle {vehicle_id} removed from queue"}
        
    except Exception as e:
        logging.error(f"Error removing vehicle from queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@router.get("/vehicle/{vehicle_id}/history")
async def get_vehicle_history_endpoint(vehicle_id: str):
    history = await get_vehicle_history(vehicle_id)
    return {"vehicle_id": vehicle_id, "history": history}

@router.get("/statistics/trips")
async def get_statistics(
    start_date: datetime = None,
    end_date: datetime = None,
    vehicle_type: str = None,
    vehicle_shift: str = None
):
    stats = await get_trip_statistics(start_date, end_date, vehicle_type, vehicle_shift)
    return stats