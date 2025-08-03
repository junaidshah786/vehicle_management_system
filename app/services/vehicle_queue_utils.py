import logging
from firebase_admin import firestore
from datetime import datetime
from app.services.firebase import db
from typing import List, Dict, Any


def get_queue_entry(vehicle_id: str):
    results = db.collection("vehicleQueue").where("vehicle_id", "==", vehicle_id).limit(1).stream()
    for doc in results:
        return doc.id, doc.to_dict()
    return None, None


def release_vehicle_from_queue(doc_id: str, vehicle_type: str, current_rank: int):
    queue_ref = db.collection("vehicleQueue")

    # Delete the current entry
    queue_ref.document(doc_id).delete()

    # Update lower ranked vehicles
    lower_ranked_query = queue_ref \
        .where("vehicle_type", "==", vehicle_type) \
        .where("queue_rank", ">", current_rank) \
        .stream()

    batch = db.batch()
    for doc in lower_ranked_query:
        doc_ref = queue_ref.document(doc.id)
        doc_data = doc.to_dict()
        updated_rank = doc_data["queue_rank"] - 1
        batch.update(doc_ref, {"queue_rank": updated_rank})
    batch.commit()


def fetch_vehicle_details(vehicle_id: str):
    doc = db.collection("vehicleDetails").document(vehicle_id).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def get_next_queue_rank(vehicle_type: str) -> int:
    query = db.collection("vehicleQueue") \
              .where("vehicle_type", "==", vehicle_type) \
              .order_by("queue_rank", direction=firestore.Query.DESCENDING) \
              .limit(1).stream()

    for doc in query:
        return doc.to_dict().get("queue_rank", 0) + 1
    return 1


def add_vehicle_to_queue(vehicle_id: str, registration_number: str, vehicle_type: str, username: str) -> int:
    next_rank = get_next_queue_rank(vehicle_type)

    queue_entry = {
        "vehicle_id": vehicle_id,
        "registration_number": registration_number,
        "vehicle_type": vehicle_type,
        "username": username,
        "queue_rank": next_rank,
        "status": "waiting",
        "added_at": datetime.utcnow()
    }

    db.collection("vehicleQueue").add(queue_entry)
    return next_rank


def log_queue_history(vehicle_id: str, action: str, queue_rank: int, vehicle_type: str, username: str):
    history_entry = {
        "vehicle_id": vehicle_id,
        "action": action,  # "added" or "removed"
        "queue_rank": queue_rank,
        "vehicle_type": vehicle_type,
        "username": username,
        "timestamp": datetime.utcnow()
    }
    db.collection("vehicleQueueHistory").add(history_entry)



def fetch_vehicles_by_type_sorted(vehicle_type: str) -> List[Dict[str, Any]]:
    """
    Fetches all vehicles of a specific type from the queue,
    sorted by their queue_rank (ascending).
    """
    try:

        docs = db.collection("vehicleQueue").where("vehicle_type", "==", vehicle_type).stream()

        vehicles = []
        for doc in docs:
            data = doc.to_dict()
            # data["doc_id"] = doc.id
            # data.pop("vehicle", None)
            vehicles.append(data)

        vehicles.sort(key=lambda x: x.get("queue_rank", float("inf")))
        return vehicles
    except Exception as e:
        logging.error(f"Error fetching vehicles by type {vehicle_type}: {e}")
        return []
