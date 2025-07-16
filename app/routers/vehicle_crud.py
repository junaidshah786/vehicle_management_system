import logging
import traceback
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
# from passlib.context import CryptContext
from firebase_admin import credentials, firestore, initialize_app

app = FastAPI()
router = APIRouter()

cred = credentials.Certificate("./vehiclemanagementsystem-e76c4-firebase-adminsdk-fbsvc-db53f875ef.json")
initialize_app(cred)
db = firestore.client()




# Include the router in the main FastAPI app
app.include_router(router)


