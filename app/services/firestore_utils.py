from app.services.firebase import db
from fastapi import HTTPException
from app.config.config import vehicle_collection_name

from google.cloud import firestore
from typing import List, Dict, Optional



def fetch_registered_vehicle_summary(
    vehicleType: Optional[str] = None,
    seatingCapacity: Optional[int] = None,
    status: Optional[str] = None,
    vehicleShift: Optional[str] = None,  # <-- Added
    search: Optional[str] = None
) -> List[Dict]:
    vehicles_ref = db.collection(vehicle_collection_name)
    docs = vehicles_ref.stream()

    summary_list = []
    for doc in docs:
        data = doc.to_dict()
        summary = {
            "_id": data.get("_id", doc.id),
            "createdAt": data.get("createdAt"),
            "updatedAt": data.get("updatedAt"),
            "ownerName": data.get("ownerName"),
            "ownerPhone": data.get("ownerPhone"),
            "seatingCapacity": data.get("seatingCapacity"),
            "registrationNumber": data.get("registrationNumber"),
            "vehicleType": data.get("vehicleType"),
            "status": data.get("status"),
            "vehicleShift": data.get("vehicleShift"),  # <-- Added
        }

        # Apply filters
        if vehicleType and summary["vehicleType"] != vehicleType:
            continue
        if seatingCapacity and summary["seatingCapacity"] != seatingCapacity:
            continue
        if status and summary["status"] != status:
            continue
        if vehicleShift and summary["vehicleShift"] != vehicleShift:  # <-- Added
            continue

        # Apply search (case-insensitive)
        if search:
            search_lower = search.lower()
            if not (
                (summary["ownerName"] and search_lower in summary["ownerName"].lower()) or
                (summary["registrationNumber"] and search_lower in summary["registrationNumber"].lower()) or
                (summary["ownerPhone"] and search_lower in str(summary["ownerPhone"]))
            ):
                continue

        summary_list.append(summary)

    return summary_list



def fetch_vehicle_data(vehicle_id: str):
    doc_ref = db.collection(vehicle_collection_name).document(vehicle_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return doc.to_dict()


def update_vehicle(vehicle_id: str, update_data: Dict) -> bool:
    doc_ref = db.collection(vehicle_collection_name).document(vehicle_id)
    if doc_ref.get().exists:
        doc_ref.update(update_data)
        return True
    return False

def delete_vehicle(vehicle_id: str) -> bool:
    doc_ref = db.collection(vehicle_collection_name).document(vehicle_id)
    if doc_ref.get().exists:
        doc_ref.delete()
        return True
    return False
