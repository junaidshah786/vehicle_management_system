from app.routers.firebase import db
from fastapi import HTTPException
from app.config.config import vehicle_collection_name

from google.cloud import firestore
from typing import List, Dict


def fetch_registered_vehicle_summary() -> List[Dict]:
    vehicles_ref = db.collection(vehicle_collection_name)
    docs = vehicles_ref.stream()

    summary_list = []
    for doc in docs:
        data = doc.to_dict()
        summary = {
            "ownerName": data.get("ownerName"),
            "registrationNumber": data.get("registrationNumber"),
            "vehicleType": data.get("vehicleType"),
            "status": data.get("status")
        }
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
