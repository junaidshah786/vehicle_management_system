"""
Plate Detection Router
Endpoints for vehicle check-in, release, and check-out via license plate image upload.
Uses unified adapter pattern - resolves plate number then calls core logic.
WITH PROFILING - Logs time taken by each step (logs only, not in API response)
"""

import logging
import traceback
import time
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config.config import MAX_IMAGE_SIZE_MB, ACTIVE_TRIPS_COLLECTION
from app.services.plate_detection import plate_detection_service, PlateDetectionResult
from app.services.firebase import db
from app.services.vehicle_queue_utils import (
    add_vehicle_to_queue_firestore,
    get_vehicle_details_firestore,
    get_vehicle_from_queue,
    log_queue_history_firestore,
    release_vehicle_from_queue_firestore,
    update_queue_ranks_after_removal
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ============ HELPER FUNCTIONS ============

async def get_vehicle_by_registration(registration_number: str) -> Optional[dict]:
    """Find vehicle by registration number (adapter function)."""
    try:
        vehicles_ref = db.collection("vehicleDetails")
        clean_reg = registration_number.replace(" ", "").upper()
        
        query = vehicles_ref.where("registrationNumber", "==", clean_reg)
        docs = list(query.stream())
        
        if docs:
            vehicle_data = docs[0].to_dict()
            vehicle_data["_id"] = docs[0].id
            return vehicle_data
        
        query = vehicles_ref.where("registrationNumber", "==", registration_number)
        docs = list(query.stream())
        
        if docs:
            vehicle_data = docs[0].to_dict()
            vehicle_data["_id"] = docs[0].id
            return vehicle_data
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding vehicle by registration: {e}")
        return None


async def get_active_trip(vehicle_id: str) -> Optional[dict]:
    """Get vehicle from active trips collection"""
    try:
        doc = db.collection(ACTIVE_TRIPS_COLLECTION).document(vehicle_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"Error fetching active trip: {e}")
        return None


async def move_to_active_trips(vehicle_id: str, queue_data: dict, released_by: str):
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
        
    except Exception as e:
        logger.error(f"Error moving to active trips: {e}")
        raise


async def remove_from_active_trips(vehicle_id: str):
    """Remove vehicle from active trips collection"""
    try:
        db.collection(ACTIVE_TRIPS_COLLECTION).document(vehicle_id).delete()
    except Exception as e:
        logger.error(f"Error removing from active trips: {e}")
        raise


def validate_image_size(file: UploadFile):
    """Validate uploaded image size"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file."
        )


async def detect_plate_from_upload(file: UploadFile) -> PlateDetectionResult:
    """Common plate detection logic for all endpoints."""
    validate_image_size(file)
    
    image_bytes = await file.read()
    
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum size is {MAX_IMAGE_SIZE_MB}MB."
        )
    
    result = plate_detection_service.detect_plate(image_bytes)
    
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "plate_detection_failed",
                "message": result.error_message,
                "raw_text": result.raw_text,
                "confidence": result.confidence
            }
        )
    
    return result


# ============ DETECT ONLY (NO ACTION) ============

@router.post("/vehicle/detect-plate")
async def detect_plate_only(
    file: UploadFile = File(..., description="Image of vehicle license plate")
):
    """Detect license plate from image without performing any action."""
    try:
        total_start = time.perf_counter()
        
        detection = await detect_plate_from_upload(file)
        
        db_start = time.perf_counter()
        vehicle = await get_vehicle_by_registration(detection.plate_number)
        db_time_ms = (time.perf_counter() - db_start) * 1000
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # LOG TIMING PROFILE (logs only)
        logger.info(
            f"DETECT-ONLY | YOLO: {detection.yolo_time_ms:.1f}ms | "
            f"OCR: {detection.ocr_time_ms:.1f}ms | DB: {db_time_ms:.1f}ms | "
            f"Total: {total_time_ms:.1f}ms | Plate: {detection.plate_number}"
        )
        
        return {
            "detected_plate": detection.plate_number,
            "confidence": detection.confidence,
            "is_registered": vehicle is not None,
            "vehicle_info": {
                "vehicleId": vehicle.get("_id"),
                "ownerName": vehicle.get("ownerName"),
                "vehicleType": vehicle.get("vehicleType"),
                "vehicleShift": vehicle.get("vehicleShift")
            } if vehicle else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_plate_only: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to detect plate: {str(e)}")


# ============ CHECK-IN VIA PLATE ============

@router.post("/vehicle/detect-plate/check-in")
async def check_in_via_plate(
    file: UploadFile = File(..., description="Image of vehicle license plate"),
    name: Optional[str] = Form(None, description="Driver name"),
    contact: Optional[str] = Form(None, description="Driver contact")
):
    """Check-in a vehicle using license plate image."""
    try:
        total_start = time.perf_counter()
        
        detection = await detect_plate_from_upload(file)
        plate_number = detection.plate_number
        
        db_start = time.perf_counter()
        vehicle = await get_vehicle_by_registration(plate_number)
        db_time_ms = (time.perf_counter() - db_start) * 1000
        
        if not vehicle:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "vehicle_not_found",
                    "message": f"Vehicle with plate {plate_number} is not registered in the system.",
                    "detected_plate": plate_number
                }
            )
        
        vehicle_id = vehicle["_id"]
        registration_number = vehicle.get("registrationNumber", plate_number)
        vehicle_type = vehicle.get("vehicleType")
        vehicle_shift = vehicle.get("vehicleShift")
        
        existing_entry = await get_vehicle_from_queue(vehicle_id)
        if existing_entry:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "already_in_queue",
                    "message": f"Vehicle already in queue at position {existing_entry.get('queue_rank')}",
                    "detected_plate": plate_number
                }
            )
        
        active_trip = await get_active_trip(vehicle_id)
        if active_trip:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "on_active_trip",
                    "message": "Vehicle is currently on a trip. Please check-out first.",
                    "detected_plate": plate_number
                }
            )
        
        driver_name = name or vehicle.get("ownerName", "N/A")
        
        next_rank = await add_vehicle_to_queue_firestore(
            vehicle_id,
            registration_number,
            vehicle_type,
            driver_name,
            vehicle_shift
        )
        
        await log_queue_history_firestore(
            vehicle_id,
            "checked_in",
            next_rank,
            vehicle_type,
            driver_name,
            checkin_time=datetime.utcnow(),
            checkout_time=None
        )
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # LOG TIMING PROFILE (logs only)
        logger.info(
            f"CHECK-IN | {plate_number} Rank {next_rank} | "
            f"YOLO: {detection.yolo_time_ms:.1f}ms | OCR: {detection.ocr_time_ms:.1f}ms | "
            f"DB: {db_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms"
        )
        
        return {
            "message": "Vehicle checked in successfully",
            "detected_plate": plate_number,
            "confidence": detection.confidence,
            "action_result": {
                "vehicleId": vehicle_id,
                "registrationNumber": registration_number,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "queueRank": next_rank
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_in_via_plate: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to check in vehicle: {str(e)}")


# ============ RELEASE VIA PLATE ============

@router.post("/vehicle/detect-plate/release")
async def release_via_plate(
    file: UploadFile = File(..., description="Image of vehicle license plate"),
    name: Optional[str] = Form(None, description="Person releasing the vehicle")
):
    """Release a vehicle from queue using license plate image."""
    try:
        total_start = time.perf_counter()
        
        detection = await detect_plate_from_upload(file)
        plate_number = detection.plate_number
        
        db_start = time.perf_counter()
        vehicle = await get_vehicle_by_registration(plate_number)
        db_time_ms = (time.perf_counter() - db_start) * 1000
        
        if not vehicle:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "vehicle_not_found",
                    "message": f"Vehicle with plate {plate_number} is not registered.",
                    "detected_plate": plate_number
                }
            )
        
        vehicle_id = vehicle["_id"]
        
        existing_entry = await get_vehicle_from_queue(vehicle_id)
        if not existing_entry:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_in_queue",
                    "message": "Vehicle not found in queue.",
                    "detected_plate": plate_number
                }
            )
        
        current_rank = existing_entry.get("queue_rank")
        vehicle_type = existing_entry.get("vehicle_type")
        vehicle_shift = existing_entry.get("vehicleShift")
        registration_number = existing_entry.get("registration_number")
        
        if current_rank != 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_first_in_queue",
                    "message": f"Only vehicle at position 1 can be released. Current position: {current_rank}",
                    "detected_plate": plate_number,
                    "current_rank": current_rank
                }
            )
        
        released_by = name or "N/A"
        await move_to_active_trips(vehicle_id, existing_entry, released_by)
        await release_vehicle_from_queue_firestore(vehicle_id)
        await update_queue_ranks_after_removal(current_rank, vehicle_type)
        
        await log_queue_history_firestore(
            vehicle_id,
            "released",
            current_rank,
            vehicle_type,
            released_by,
            checkin_time=existing_entry.get("queued_at"),
            checkout_time=None,
            released_time=datetime.utcnow()
        )
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # LOG TIMING PROFILE (logs only)
        logger.info(
            f"RELEASE | {plate_number} | "
            f"YOLO: {detection.yolo_time_ms:.1f}ms | OCR: {detection.ocr_time_ms:.1f}ms | "
            f"DB: {db_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms"
        )
        
        return {
            "message": "Vehicle released for trip",
            "detected_plate": plate_number,
            "confidence": detection.confidence,
            "action_result": {
                "vehicleId": vehicle_id,
                "registrationNumber": registration_number,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "previousQueueRank": current_rank,
                "status": "on_trip"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in release_via_plate: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to release vehicle: {str(e)}")


# ============ CHECK-OUT VIA PLATE ============

@router.post("/vehicle/detect-plate/check-out")
async def check_out_via_plate(
    file: UploadFile = File(..., description="Image of vehicle license plate"),
    name: Optional[str] = Form(None, description="Person checking out the vehicle")
):
    """Check-out a vehicle from active trip using license plate image."""
    try:
        total_start = time.perf_counter()
        
        detection = await detect_plate_from_upload(file)
        plate_number = detection.plate_number
        
        db_start = time.perf_counter()
        vehicle = await get_vehicle_by_registration(plate_number)
        db_time_ms = (time.perf_counter() - db_start) * 1000
        
        if not vehicle:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "vehicle_not_found",
                    "message": f"Vehicle with plate {plate_number} is not registered.",
                    "detected_plate": plate_number
                }
            )
        
        vehicle_id = vehicle["_id"]
        
        active_trip = await get_active_trip(vehicle_id)
        if not active_trip:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_on_trip",
                    "message": "Vehicle not found in active trips. Cannot check out.",
                    "detected_plate": plate_number
                }
            )
        
        vehicle_type = active_trip.get("vehicleType")
        vehicle_shift = active_trip.get("vehicleShift")
        registration_number = active_trip.get("registrationNumber")
        queued_at = active_trip.get("queuedAt")
        released_at = active_trip.get("releasedAt")
        
        await remove_from_active_trips(vehicle_id)
        
        checkout_by = name or "N/A"
        await log_queue_history_firestore(
            vehicle_id,
            "checked_out",
            active_trip.get("queueRank"),
            vehicle_type,
            checkout_by,
            checkin_time=queued_at,
            checkout_time=datetime.utcnow(),
            released_time=released_at
        )
        
        trip_duration = None
        if released_at:
            released_at_naive = released_at.replace(tzinfo=None) if hasattr(released_at, 'tzinfo') and released_at.tzinfo else released_at
            trip_duration = str(datetime.utcnow() - released_at_naive)
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # LOG TIMING PROFILE (logs only)
        logger.info(
            f"CHECK-OUT | {plate_number} Duration: {trip_duration} | "
            f"YOLO: {detection.yolo_time_ms:.1f}ms | OCR: {detection.ocr_time_ms:.1f}ms | "
            f"DB: {db_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms"
        )
        
        return {
            "message": "Vehicle checked out successfully",
            "detected_plate": plate_number,
            "confidence": detection.confidence,
            "action_result": {
                "vehicleId": vehicle_id,
                "registrationNumber": registration_number,
                "vehicleType": vehicle_type,
                "vehicleShift": vehicle_shift,
                "tripDuration": trip_duration
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_out_via_plate: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to check out vehicle: {str(e)}")
