import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter

from ..services.login import create_access_token
from app.services.pydantic import LoginRequest, SignupRequest
from app.services.firebase import db

app = FastAPI()
router = APIRouter()

# from jose import jwt

# SECRET_KEY = "your-secret-key"
# ALGORITHM = "HS256"

# def create_access_token(data: dict):
#     return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


# 🔐 Signup API (admin or user)
@router.post("/signup", tags=["Authentication"])
async def signup(signup_details: SignupRequest):
    try:
        logging.info(f"Signup attempt for {signup_details.username}")

        existing_user = db.collection("users").where("username", "==", signup_details.username).get()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

        db.collection("users").add({
            "username": signup_details.username,
            "name": signup_details.name,
            "contact": signup_details.contact,
            "passwordHash": signup_details.passwordHash,
            "role": "user"
        })

        return {"message": "Signup successful"}
    except Exception:
        logging.error(f"Error during signup: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 🔐 Unified Login for all roles
@router.post("/login", tags=["Authentication"])
async def login(login_details: LoginRequest):
    try:
        logging.info(f"Login attempt for {login_details.username}")

        user_docs = db.collection("users").where("username", "==", login_details.username).get()
        if not user_docs:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user_data = user_docs[0].to_dict()
        if login_details.passwordHash != user_data["passwordHash"]:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # token = create_access_token({"sub": login_details.username, "role": user_data["role"]})

        return {
            "accessToken": "token_placeholder",  # Replace with actual token generation logic
            "username": user_data.get("name", "Unknown User"),
            "role": user_data["role"],
            "tokenType": "bearer",
            "message": f"Login successful as {user_data['role']}"
        }

    except Exception as e:
        logging.error(f"Error during login: {traceback.format_exc()}")
        raise HTTPException(status_code= e.status_code if hasattr(e, 'status_code') else 500, detail=str(e))

# Include the router in the main FastAPI app
app.include_router(router)


