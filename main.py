import logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import authentication, queue_management, reporting, vehicle_crud

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(authentication.router, prefix="/v1", tags=["Authentication"])
app.include_router(vehicle_crud.router, prefix="/v1", tags=[" Vehicles"])
app.include_router(queue_management.router, prefix="/v1", tags=["Queue Management"])
app.include_router(reporting.router, prefix="/v1", tags=["Reporting"])




