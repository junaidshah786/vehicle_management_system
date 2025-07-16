import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter, Path
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
from app.routers.firebase import db
from app.routers.firestore_utils import delete_vehicle, fetch_registered_vehicle_summary, fetch_vehicle_data, update_vehicle
from app.services.pdf_generator import generate_vehicle_icard_pdf
from app.services.pydantic import VehicleRegistration, VehicleUpdateRequest

app = FastAPI()
router = APIRouter()

# 1. Vehicle Registration API
@router.get("/fetch-vehicles", summary="Fetch all registered vehicles (summary)")
async def get_registered_vehicles():
    try:
        vehicles = fetch_registered_vehicle_summary()
        return {"status": "success", "count": len(vehicles), "data": vehicles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching vehicles: {e}")
    
# 2. Vehicle Registration API
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

# 3. Download Vehicle I-Card API
@router.get("/download-icard/{vehicle_id}")
def download_icard(vehicle_id: str):
    vehicle = fetch_vehicle_data(vehicle_id)
    pdf_buffer = generate_vehicle_icard_pdf(vehicle_id, vehicle)

    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=vehicle_icard_{vehicle_id}.pdf"
    })

# 4. Update Vehicle API
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

# 5. Delete Vehicle API
@router.delete("/vehicles/{vehicle_id}", summary="Delete a registered vehicle")
async def remove_vehicle(vehicle_id: str = Path(..., description="Firestore document ID")):
    success = delete_vehicle(vehicle_id)
    if not success:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"status": "success", "message": "Vehicle deleted successfully"}


# Include the router in the main FastAPI app
app.include_router(router)


