import datetime
import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from passlib.context import CryptContext
from app.routers.firestore_utils import fetch_vehicle_data
from app.services.pdf_generator import generate_vehicle_icard_pdf
from firebase_admin import credentials, firestore, initialize_app

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

@router.post("/vehciles")
def register_vehicle(data: VehicleRegistration):
    try:
        vehicle_data = data.dict()
        now = datetime.datetime.utcnow().isoformat() + "Z"

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


@router.get("/download-i-card/{vehicle_id}")
def download_icard(vehicle_id: str):
    vehicle = fetch_vehicle_data(vehicle_id)
    pdf_buffer = generate_vehicle_icard_pdf(vehicle_id, vehicle)

    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=vehicle_icard_{vehicle_id}.pdf"
    })
# Include the router in the main FastAPI app
app.include_router(router)


