import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

from app.routers.firestore_utils import delete_vehicle, fetch_registered_vehicle_summary, fetch_vehicle_data, update_vehicle
from app.services.pdf_generator import generate_vehicle_icard_pdf
from firebase_admin import firestore

app = FastAPI()
router = APIRouter()

# cred = credentials.Certificate("./vehiclemanagementsystem-e76c4-firebase-adminsdk-fbsvc-db53f875ef.json")
# initialize_app(cred)
db = firestore.client()
from pydantic import BaseModel, Field
from enum import Enum

# Enum for vehicle types
class VehicleTypeEnum(str, Enum):
    innova = "innova"
    tavera = "tavera"
    sumo = "sumo"
    xylo = "xylo"
    scorpio = "scorpio"
    bolero = "bolero"
    other = "other"

# Simplified model with inline constraints
class VehicleRegistration(BaseModel):
    registrationNumber: str = Field(..., min_length=6, max_length=15, description="e.g., JK01AB1234")
    vehicleType: VehicleTypeEnum
    ownerName: str = Field(..., min_length=3, max_length=50)
    ownerPhone: str = Field(..., pattern=r"^\+91-\d{10}$")
    seatingCapacity: int = Field(..., gt=0, lt=100)
    status: str = Field(default="active", pattern="^(active|inactive)$")


@router.get("/fetch-vehicles", summary="Fetch all registered vehicles (summary)")
async def get_registered_vehicles():
    try:
        vehicles = fetch_registered_vehicle_summary()
        return {"status": "success", "count": len(vehicles), "data": vehicles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching vehicles: {e}")
    
    
@router.post("/vehicles")
def register_vehicle(data: VehicleRegistration):
    try:
        vehicle_data = data.dict()
        now = datetime.utcnow().isoformat() + "Z"

        # Assign a Firestore document ID (e.g., auto-generated or reg no)
        doc_ref = db.collection("vehicleDetails").document()

        # Prepare data with metadata
        vehicle_data.update({
            "_id": doc_ref.id,
            "createdAt": now,
            "updatedAt": now
        })

        doc_ref.set(vehicle_data)
        return {"message": "Vehicle registered successfully", "vehicleId": doc_ref.id}
    except Exception as e:
        logging.error(f"Error registering vehicle: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-icard/{vehicle_id}")
def download_icard(vehicle_id: str):
    vehicle = fetch_vehicle_data(vehicle_id)
    pdf_buffer = generate_vehicle_icard_pdf(vehicle_id, vehicle)

    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=vehicle_icard_{vehicle_id}.pdf"
    })

class VehicleStatus(str, Enum):
    active = "active"
    inactive = "inactive"

# 2. Update Request Body with validation
class VehicleUpdateRequest(BaseModel):
    ownerName: Optional[str] = None
    registrationNumber: Optional[str] = None
    vehicleType: Optional[str] = None
    status: Optional[VehicleStatus] = None  # Only active/inactive allowed
    seatingCapacity: Optional[int] = None
    ownerPhone: Optional[str] = None

# 3. Update Vehicle API
@router.put("/vehicles/{vehicle_id}", summary="Edit a registered vehicle")
async def edit_vehicle(
    vehicle_id: str = Path(..., description="Firestore document ID"),
    update_data: VehicleUpdateRequest = ...
):
    update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
    update_dict["updatedAt"] = datetime.now(timezone.utc).isoformat()

    success = update_vehicle(vehicle_id, update_dict)
    if not success:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    return {"status": "success", "message": "Vehicle updated successfully"}


@router.delete("/vehicles/{vehicle_id}", summary="Delete a registered vehicle")
async def remove_vehicle(vehicle_id: str = Path(..., description="Firestore document ID")):
    success = delete_vehicle(vehicle_id)
    if not success:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"status": "success", "message": "Vehicle deleted successfully"}


# Include the router in the main FastAPI app
app.include_router(router)


