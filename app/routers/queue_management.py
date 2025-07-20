import logging
import traceback
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from firebase_admin import  firestore
from app.routers.firebase import db
from app.services.pydantic import QRRequest
from datetime import datetime


app = FastAPI()
router = APIRouter()

@router.post("/verify-and-queue-vehicle")
async def verify_and_add_to_queue(request: QRRequest):
    try:
        # Step 1: Decode QR
        if "_" not in request.qr_data:
            raise HTTPException(status_code=400, detail="Invalid QR format")
        
        vehicle_id, registration_number = request.qr_data.split("_", 1)

        # Step 2: Fetch vehicle data from Firestore
        vehicle_doc = db.collection("vehicleDetails").document(vehicle_id).get()
        if not vehicle_doc.exists:
            raise HTTPException(status_code=404, detail="Vehicle not found")

        vehicle_data = vehicle_doc.to_dict()

        # Optional: double-check registration number
        if vehicle_data.get("registrationNumber") != registration_number:
            raise HTTPException(status_code=403, detail="Registration number mismatch")

        vehicle_type = vehicle_data.get("vehicleType")
        if not vehicle_type:
            raise HTTPException(status_code=400, detail="Vehicle type missing")

        # Step 3: Get current max queue rank for this vehicleType
        queue_ref = db.collection("vehicleQueue")
        query = queue_ref.where("vehicle_type", "==", vehicle_type).order_by("queue_rank", direction=firestore.Query.DESCENDING).limit(1)
        results = query.stream()

        max_rank = 0
        for doc in results:
            max_rank = doc.to_dict().get("queue_rank", 0)
            break

        next_rank = max_rank + 1

        # Step 4: Add to queue
        queue_entry = {
            "vehicle_id": vehicle_id,
            "registration_number": registration_number,
            "vehicle_type": vehicle_type,
            "username": request.username,
            "queue_rank": next_rank,
            "status": "waiting",
            "added_at": datetime.utcnow()
        }

        db.collection("vehicleQueue").add(queue_entry)

        return {
            "message": "Vehicle added to queue successfully",
            "vehicleId": vehicle_id,
            "queueRank": next_rank,
            "vehicleType": vehicle_type
        }

    except Exception as e:
        logging.error(f"Error in verify_and_add_to_queue: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    

# Include the router in the main FastAPI app
app.include_router(router)