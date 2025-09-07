import io
import logging
import traceback
from datetime import datetime, timedelta
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.config.config import QUEUE_HISTORY_COLLECTION
from app.services.firebase import db
from app.services.pydantic import LoginRequest, SignupRequest

router = APIRouter()


@router.get("/queue-history-report")
async def queue_history_report(
    start_date: str = Query(..., description="Report start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="Report end date (YYYY-MM-DD)"),
    vehicle_type: str = Query(None, description="Vehicle type to filter (optional)"),
    vehicle_shift: str = Query(None, description="Vehicle shift to filter (optional)")
):
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_dt = end_dt + timedelta(days=1)  # Include end date fully

        # Query Firestore for queue history with checkout_time present
        query = (
            db.collection(QUEUE_HISTORY_COLLECTION)
            .where("timestamp", ">=", start_dt)
            .where("timestamp", "<", end_dt)
            .where("checkout_time", "!=", None)  # Only fetch docs with checkout_time
        )
        if vehicle_type:
            query = query.where("vehicle_type", "==", vehicle_type)
        if vehicle_shift:
            query = query.where("vehicleShift", "==", vehicle_shift)  # <-- Added

        docs = query.stream()

        # Collect data
        records = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure timestamp is a datetime object
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            records.append(data)

        if not records:
            return {"message": "No data found for given filters"}

        # Group by day
        df = pd.DataFrame(records)
        df["date"] = df["timestamp"].dt.date
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        if "checkin_time" in df.columns:
            df["checkin_time"] = pd.to_datetime(df["checkin_time"]).dt.tz_localize(None)
        if "checkout_time" in df.columns:
            df["checkout_time"] = pd.to_datetime(df["checkout_time"]).dt.tz_localize(None)

        # Create Excel file with each day as a sheet
        output = io.BytesIO()
        with pd.ExcelWriter(output) as writer:
            for day, group in df.groupby("date"):
                group = group.drop(columns=["date", "queue_rank", "vehicle_id", "action"], errors="ignore")
                group.to_excel(writer, sheet_name=str(day), index=False)
        output.seek(0)

        filename = f"queue_history_{start_date}_to_{end_date}.xlsx"
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logging.error(f"Error generating report: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to generate report")


# Include the router in the main FastAPI app


