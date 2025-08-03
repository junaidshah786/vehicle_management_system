from fastapi import APIRouter, HTTPException
from app.services.vehicle_queue_utils import (
    get_queue_entry,
    release_vehicle_from_queue,
    fetch_vehicle_details,
    add_vehicle_to_queue,
    log_queue_history
)
from app.services.pydantic import QRRequest
import logging
import traceback

router = APIRouter()


from fastapi import Query
from typing import  Dict, Any


from fastapi import Query
from typing import Dict, Any
from app.services.vehicle_queue_utils import fetch_vehicles_by_type_sorted


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
        # Check if vehicle already in queue
        doc_id, existing_entry = get_queue_entry(vehicle_id)

        if existing_entry:
            current_rank = existing_entry.get("queue_rank")
            vehicle_type = existing_entry.get("vehicle_type")

            # 🚫 Reject if rank is not 1
            if current_rank != 1:
                raise HTTPException(
                    status_code=409,
                    detail="Only vehicle on TOP (rank 1) can be released"
                )

            # ✅ Proceed to release
            release_vehicle_from_queue(doc_id, vehicle_type, current_rank)

            # Log history
            log_queue_history(vehicle_id, "removed", current_rank, vehicle_type, request.username)

            return {
                "message": "Vehicle released from queue",
                "vehicleId": vehicle_id,
                "vehicleType": vehicle_type,
                "previousQueueRank": current_rank
            }


        # Not in queue — fetch details
        vehicle_data = fetch_vehicle_details(vehicle_id)
        if not vehicle_data:
            logging.error(f"Vehicle with ID {vehicle_id} not found")
            raise HTTPException(status_code=404, detail="Vehicle not found")

        if vehicle_data.get("registrationNumber") != registration_number:
            logging.error(f"Registration number mismatch: expected {vehicle_data.get('registrationNumber')}, got {registration_number}")
            raise HTTPException(status_code=403, detail="Registration number mismatch")

        vehicle_type = vehicle_data.get("vehicleType")
        if not vehicle_type:
            logging.error(f"Vehicle type missing for vehicle ID {vehicle_id}")
            raise HTTPException(status_code=400, detail="Vehicle type missing")

        # Add to queue
        next_rank = add_vehicle_to_queue(vehicle_id, registration_number, vehicle_type, request.name)

        # Log history
        log_queue_history(vehicle_id, "added", next_rank, vehicle_type, request.name)

        return {
            "message": "Vehicle added to queue successfully",
            "vehicleId": vehicle_id,
            "queueRank": next_rank,
            "vehicleType": vehicle_type
        }

    except Exception as e:
        logging.error(f"Error in verify_and_add_to_queue: {traceback.format_exc()}")
        raise HTTPException(status_code= e.status_code if hasattr(e, 'status_code') else 500, detail=str(e))
