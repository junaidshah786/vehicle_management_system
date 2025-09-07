import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter, Path
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
from app.services.firebase import db
from app.services.firestore_utils import delete_vehicle, fetch_registered_vehicle_summary, fetch_vehicle_data, update_vehicle
from app.services.pdf_generator import generate_vehicle_icard_pdf
from app.services.pydantic import VehicleRegistration, VehicleUpdateRequest
from fastapi import Query
from typing import Optional

app = FastAPI()
router = APIRouter()


@router.get("/fetch-vehicles", summary="Fetch all registered vehicles with pagination, filter, and search")
async def get_registered_vehicles(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    vehicleType: Optional[str] = Query(None),
    seatingCapacity: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    vehicleShift: Optional[str] = Query(None, description="Vehicle shift (morning/day/night)"),  # <-- Added
    search: Optional[str] = Query(None, description="Search by ownerName, registrationNumber, or phone")
):
    try:
        vehicles = fetch_registered_vehicle_summary(
            vehicleType=vehicleType,
            seatingCapacity=seatingCapacity,
            status=status,
            vehicleShift=vehicleShift,  # <-- Added
            search=search
        )

        # Pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_data = vehicles[start:end]

        return {
            "status": "success",
            "page": page,
            "limit": limit,
            "total": len(vehicles),
            "data": paginated_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching vehicles: {e}")

# 1. Vehicle Registration API
@router.post("/vehicles")
def register_vehicle(data: VehicleRegistration):
    try:
        vehicle_data = data.dict()
        now = datetime.utcnow().isoformat() + "Z"

        # Check if a vehicle with the same registrationNumber already exists
        existing_query = (
            db.collection("vehicleDetails")
            .where("registrationNumber", "==", vehicle_data["registrationNumber"])
            .limit(1)
            .stream()
        )
        if any(existing_query):
            raise HTTPException(status_code=400, detail="Vehicle with this registration number already exists")

        # Assign Firestore document ID
        doc_ref = db.collection("vehicleDetails").document()

        # Add metadata
        vehicle_data.update({
            "_id": doc_ref.id,
            "createdAt": now,
            "updatedAt": now,
            "vehicleShift": vehicle_data.get("vehicleShift"),  # <-- Added
        })

        doc_ref.set(vehicle_data)
        return {"message": "Vehicle registered successfully", "vehicleId": doc_ref.id}

    except HTTPException:
        raise  # re-raise HTTP errors as-is
    except Exception as e:
        logging.error(f"Error registering vehicle: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")


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


