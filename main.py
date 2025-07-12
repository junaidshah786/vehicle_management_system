import logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import authentication

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(authentication.router, prefix="/v1/login", tags=["Authentication"])






