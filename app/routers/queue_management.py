from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.config.config import QUEUE_HISTORY_COLLECTION, VEHICLE_QUEUE_COLLECTION, VEHICLES_COLLECTION
from app.services.pydantic import QRRequest, RegisterDeviceRequest, TestFcmNotificationRequest, UnregisterDeviceRequest
import logging
import traceback
from app.services.firebase import db
from fastapi import Query
from typing import Optional, Dict, List, Any
from firebase_admin import firestore, messaging
import asyncio
from app.services.vehicle_queue_utils import add_vehicle_to_queue_firestore, cleanup_invalid_tokens, fetch_vehicles_by_type_sorted, get_vehicle_details_firestore, get_vehicle_from_queue, log_queue_history_firestore, release_vehicle_from_queue_firestore, update_queue_ranks_after_removal
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



async def get_active_device_tokens() -> List[str]:
    """Fetch all active device FCM tokens from Firestore"""
    try:
        active_devices_ref = db.collection("activeDevices")
        docs = active_devices_ref.stream()
        
        tokens = []
        for doc in docs:
            device_data = doc.to_dict()
            token = device_data.get("fcm_token")
            is_active = device_data.get("is_active", False)
            
            if token and is_active:
                tokens.append(token)
        
        logging.info(f"Retrieved {len(tokens)} active device tokens")
        return tokens
        
    except Exception as e:
        logging.error(f"Error fetching active device tokens: {e}")
        return []

async def send_fcm_notification_to_active_devices(title: str, body: str, data: Dict = None):
    """Send FCM notification to all active devices - Individual method only"""
    try:
        # Get active device tokens
        tokens = await get_active_device_tokens()
        
        if not tokens:
            logging.warning("No active device tokens found")
            return
        
        # Prepare data payload (optional)
        notification_data = data or {}
        notification_data.update({
            "action": "refresh_queue",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Convert all data values to strings (FCM requirement)
        string_data = {k: str(v) for k, v in notification_data.items()}
        
        successful_sends = 0
        failed_tokens = []
        
        logging.info(f"Attempting to send FCM notifications to {len(tokens)} devices")
        
        # Send to each token individually
        for i, token in enumerate(tokens):
            try:
                # Create individual message
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body
                    ),
                    data=string_data,
                    token=token
                )
                
                # Send message synchronously in thread
                def send_single_message():
                    try:
                        return messaging.send(message)
                    except Exception as e:
                        # Log the specific error for this token
                        logging.error(f"Firebase messaging.send error for token {token[:10]}...: {str(e)}")
                        raise e
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, send_single_message
                )
                
                successful_sends += 1
                logging.debug(f"Successfully sent FCM to token {i+1}/{len(tokens)}: {token[:10]}...")
                
            except Exception as token_error:
                failed_tokens.append(token)
                error_msg = str(token_error)
                
                # Check if it's an authentication or configuration error
                if "404" in error_msg:
                    logging.error(f"FCM 404 Error - Possible Firebase configuration issue: {error_msg}")
                elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logging.error(f"FCM Authentication Error - Check Firebase credentials: {error_msg}")
                elif "invalid" in error_msg.lower() or "not registered" in error_msg.lower():
                    logging.warning(f"Invalid FCM token {token[:10]}...: {error_msg}")
                else:
                    logging.warning(f"Failed to send to token {token[:10]}...: {error_msg}")
        
        logging.info(f"FCM notifications completed: {successful_sends} successful, {len(failed_tokens)} failed out of {len(tokens)} total")
        
        # Clean up invalid tokens (only if they seem to be token-specific issues)
        if failed_tokens:
            # Only cleanup if errors seem to be invalid token related, not configuration issues
            token_specific_errors = [t for t in failed_tokens if len(failed_tokens) < len(tokens)]
            if token_specific_errors:
                await cleanup_invalid_tokens(token_specific_errors)
                logging.info(f"Cleaned up {len(token_specific_errors)} invalid tokens")
        
        return {
            "total_tokens": len(tokens),
            "successful": successful_sends,
            "failed": len(failed_tokens)
        }
        
    except Exception as e:
        logging.error(f"Error in send_fcm_notification_to_active_devices: {e}")
        logging.error(f"Full error details: {traceback.format_exc()}")
        return {
            "total_tokens": 0,
            "successful": 0,
            "failed": 0,
            "error": str(e)
        }



# # Your existing functions (keeping them as they are)
# async def get_vehicle_from_queue(vehicle_id: str) -> Optional[Dict]:
#     """Get vehicle from Firestore queue"""
#     try:
#         doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
#         doc = doc_ref.get()
#         return doc.to_dict() if doc.exists else None
#     except Exception as e:
#         logging.error(f"Error getting vehicle from queue: {e}")
#         return None

# async def get_vehicle_details_firestore(vehicle_id: str) -> Optional[Dict]:
#     """Get vehicle details from Firestore vehicles collection"""
#     try:
#         doc_ref = db.collection(VEHICLES_COLLECTION).document(vehicle_id)
#         doc = doc_ref.get()
#         return doc.to_dict() if doc.exists else None
#     except Exception as e:
#         logging.error(f"Error getting vehicle details: {e}")
#         return None

# async def add_vehicle_to_queue_firestore(vehicle_id: str, registration_number: str, vehicle_type: str, driver_name: str) -> int:
#     """Add vehicle to Firestore queue - DIRECT WRITE (triggers Cloud Function)"""
#     try:
#         # Get next rank
#         next_rank = await get_next_queue_rank(vehicle_type)
        
#         # Create vehicle queue entry
#         vehicle_queue_data = {
#             "vehicle_id": vehicle_id,
#             "registration_number": registration_number,
#             "vehicle_type": vehicle_type,
#             "queue_rank": next_rank,
#             "driver_name": driver_name,
#             "status": "waiting",
#             "queued_at": datetime.utcnow(),
#             "created_by": driver_name
#         }
        
#         # DIRECT FIRESTORE WRITE - This automatically triggers Cloud Function
#         doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
#         doc_ref.set(vehicle_queue_data)
        
#         logging.info(f"Vehicle {vehicle_id} added to Firestore queue at rank {next_rank}")
#         return next_rank
        
#     except Exception as e:
#         logging.error(f"Error adding vehicle to queue: {e}")
#         raise

# async def release_vehicle_from_queue_firestore(vehicle_id: str):
#     """Remove vehicle from Firestore queue - DIRECT DELETE (triggers Cloud Function)"""
#     try:
#         # DIRECT FIRESTORE DELETE - This automatically triggers Cloud Function
#         doc_ref = db.collection(VEHICLE_QUEUE_COLLECTION).document(vehicle_id)
#         doc_ref.delete()
        
#         logging.info(f"Vehicle {vehicle_id} released from Firestore queue")
        
#     except Exception as e:
#         logging.error(f"Error releasing vehicle from queue: {e}")
#         raise

# async def get_next_queue_rank(vehicle_type: str) -> int:
#     """Get next rank for vehicle type"""
#     try:
#         # Query vehicles of same type to get max rank
#         query = db.collection(VEHICLE_QUEUE_COLLECTION)\
#                  .where("vehicle_type", "==", vehicle_type)\
#                  .order_by("queue_rank", direction=firestore.Query.DESCENDING)\
#                  .limit(1)
        
#         docs = query.stream()
#         max_rank = 0
        
#         for doc in docs:
#             max_rank = doc.to_dict().get("queue_rank", 0)
#             break
            
#         return max_rank + 1
        
#     except Exception as e:
#         logging.error(f"Error getting next queue rank: {e}")
#         return 1

# async def update_queue_ranks_after_removal(removed_rank: int, vehicle_type: str):
#     """Update ranks of vehicles after one is removed - DIRECT FIRESTORE UPDATES (triggers Cloud Function)"""
#     try:
#         # Get all vehicles with higher ranks of same type
#         query = db.collection(VEHICLE_QUEUE_COLLECTION)\
#                .where("vehicle_type", "==", vehicle_type)\
#                .where("queue_rank", ">", removed_rank)
        
#         docs = query.stream()
#         batch = db.batch()
        
#         for doc in docs:
#             current_rank = doc.to_dict().get("queue_rank")
#             new_rank = current_rank - 1
            
#             # DIRECT FIRESTORE UPDATE - This automatically triggers Cloud Function
#             batch.update(doc.reference, {"queue_rank": new_rank, "updated_at": datetime.utcnow()})
        
#         batch.commit()
#         logging.info(f"Updated ranks after removing vehicle at rank {removed_rank}")
        
#     except Exception as e:
#         logging.error(f"Error updating queue ranks: {e}")


# Enhanced endpoint with FCM notifications
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
            await log_queue_history_firestore(
                vehicle_id, "removed", current_rank, vehicle_type, request.name,
                checkin_time=existing_entry.get("queued_at"),
                checkout_time=datetime.utcnow()
            )

            # 📱 Send FCM notification for vehicle release
            await send_fcm_notification_to_active_devices(
                title="Vehicle Released from Queue",
                body=f"{vehicle_type} ({registration_number}) has been released from queue",
                data={
                    "vehicle_id": vehicle_id,
                    "vehicle_type": vehicle_type,
                    "registration_number": registration_number,
                    "action": "vehicle_released",
                    "previous_rank": str(current_rank)
                }
            )

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
        await log_queue_history_firestore(
            vehicle_id, "added", next_rank, vehicle_type, request.name,
            checkin_time=datetime.utcnow(),
            checkout_time=None
        )

        # 📱 Send FCM notification for vehicle addition
        await send_fcm_notification_to_active_devices(
            title="New Vehicle Added to Queue",
            body=f"{vehicle_type} ({registration_number}) added to queue at position {next_rank}",
            data={
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_type,
                "registration_number": registration_number,
                "action": "vehicle_added",
                "queue_rank": str(next_rank)
            }
        )

        return {
            "message": "Vehicle added to queue successfully",
            "vehicleId": vehicle_id,
            "queueRank": next_rank,
            "vehicleType": vehicle_type
        }

    except Exception as e:
        logging.error(f"Error in verify_and_add_to_queue: {traceback.format_exc()}")
        raise HTTPException(status_code= e.status_code if hasattr(e, 'status_code') else 500, detail=str(e))

# Optional: Helper endpoint to manage active devices
@router.post("/register-device")
async def register_device(device_data: RegisterDeviceRequest):
    """Register a device for FCM notifications"""
    try:
        device_id = device_data.device_id
        fcm_token = device_data.fcm_token
        username = device_data.username
        
        if not all([device_id, fcm_token]):
            raise HTTPException(status_code=400, detail="device_id and fcm_token are required")
        
        # Store in activeDevices collection
        device_doc_data = {
            "deviceId": device_id,
            "fcmToken": fcm_token,
            "username": username,
            "isActive": True,
            "lastSeen": datetime.utcnow(),
            "registeredAt": datetime.utcnow()
        }
        
        doc_ref = db.collection("activeDevices").document(device_id)
        doc_ref.set(device_doc_data, merge=True)
        
        return {"message": "Device registered successfully"}
        
    except Exception as e:
        logging.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Test endpoint with detailed error reporting
@router.post("/test-fcm")
async def test_fcm_notification(test_data: TestFcmNotificationRequest):
    """Test FCM notification functionality with detailed reporting"""
    try:
        title = test_data.title
        body = test_data.body
        
        # Send test notification
        result = await send_fcm_notification_to_active_devices(
            title=title,
            body=body,
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
        logging.error(f"Error in test FCM endpoint: {e}")
        logging.error(f"Full error: {traceback.format_exc()}")
        return {
            "message": "Test notification failed",
            "error": str(e),
            "status": "error"
        }

@router.post("/unregister-device")
async def unregister_device(device_data: UnregisterDeviceRequest):
    """Unregister a device from FCM notifications"""
    try:
        device_id = device_data.device_id
        
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id is required")
        
        # Update device as inactive
        doc_ref = db.collection("activeDevices").document(device_id)
        doc_ref.update({
            "isActive": False,
            "lastSeen": datetime.utcnow()
        })
        
        return {"message": "Device unregistered successfully"}
        
    except Exception as e:
        logging.error(f"Error unregistering device: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

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
