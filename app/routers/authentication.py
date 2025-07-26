import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter

from ..services.login import create_access_token
from app.services.pydantic import LoginRequest
from app.services.firebase import db

app = FastAPI()
router = APIRouter()



# Admin login endpoint
@router.post("/login", tags=["Authentication"])
async def user_login(login_details: LoginRequest):
    try:
        logging.info(f"Login attempt for {login_details.username}")

            # Fetch user details from Firestore
        user_docs = db.collection("admin").where("username", "==", login_details.username).get()
        
        if not user_docs:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Verify the password
        user_data = user_docs[0].to_dict()
        if login_details.passwordHash != user_data["passwordHash"]:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Create JWT token without expiry
        access_token = create_access_token(data={"sub": login_details.username})
        username = user_data.get("name", "Unknown User")
        role = user_data.get("role", "user")

        return {"accessToken": access_token, 
                "username": username,
                "role": role,
                "tokenType": "bearer",
                "message": "Login successful"}
    except Exception as e:
        logging.error(f"Error during admin login: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main FastAPI app
app.include_router(router)


