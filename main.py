# import logging
# logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from logging_config import setup_logging
import logging

# Setup logging before app starts
setup_logging()

logger = logging.getLogger(__name__)



from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import authentication, queue_management, reporting, vehicle_crud, plate_detection
from app.services.plate_cache import plate_cache
from app.services.firebase import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize plate cache with Firestore listener
    logger.info("Starting plate cache Firestore listener...")
    plate_cache.start_listener(db)
    yield
    # Shutdown: clean up listener
    plate_cache.stop_listener()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    logger.info("This message goes to both app.log and terminal!")
    return {"message": "Logging works!"}
app.include_router(authentication.router, prefix="/v1", tags=["Authentication"])
app.include_router(vehicle_crud.router, prefix="/v1", tags=[" Vehicles"])
app.include_router(queue_management.router, prefix="/v1", tags=["Queue Management"])
app.include_router(reporting.router, prefix="/v1", tags=["Reporting"])
app.include_router(plate_detection.router, prefix="/v1", tags=["Plate Detection"])




