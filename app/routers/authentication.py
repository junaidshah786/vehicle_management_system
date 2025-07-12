import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
# from passlib.context import CryptContext
from firebase_admin import credentials, firestore, initialize_app
from ..services.login import create_access_token

app = FastAPI()
router = APIRouter()

cred = credentials.Certificate("./vehiclemanagementsystem-e76c4-firebase-adminsdk-fbsvc-db53f875ef.json")
initialize_app(cred)
db = firestore.client()


class LoginRequest(BaseModel):
    username: str
    passwordHash: str

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

        return {"access_token": access_token, 
                "token_type": "bearer",
                "message": "Login successful"}
    except Exception as e:
        logging.error(f"Error during admin login: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main FastAPI app
app.include_router(router)


