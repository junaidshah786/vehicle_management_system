from app.routers.firebase import db
from fastapi import HTTPException

def fetch_vehicle_data(vehicle_id: str):
    doc_ref = db.collection("vehicleDetails").document(vehicle_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return doc.to_dict()
