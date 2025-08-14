from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


########Login Request##########
class SignupRequest(BaseModel):
    username: str
    name: str
    contact: str
    passwordHash: str

class LoginRequest(BaseModel):
    username: str
    passwordHash: str

######## Vehicle Registeration##########

# Enum for vehicle types
class VehicleTypeEnum(str, Enum):
    innova = "innova"
    tavera = "tavera"
    sumo = "sumo"
    xylo = "xylo"
    scorpio = "scorpio"
    bolero = "bolero"
    other = "other"

# Simplified model with inline constraints
class VehicleRegistration(BaseModel):
    registrationNumber: str = Field(..., min_length=6, max_length=15, description="e.g., JK01AB1234")
    vehicleType: VehicleTypeEnum
    ownerName: str = Field(..., min_length=3, max_length=50)
    ownerPhone: str = Field(..., pattern=r"^\d{10}$")
    seatingCapacity: int = Field(..., gt=0, lt=100)
    status: str = Field(default="active", pattern="^(active|inactive)$")


######## Vehicle Update Request#########

class VehicleStatus(str, Enum):
    active = "active"
    inactive = "inactive"

# 2. Update Request Body with validation
class VehicleUpdateRequest(BaseModel):
    ownerName: Optional[str] = None
    registrationNumber: Optional[str] = None
    vehicleType: Optional[str] = None
    status: Optional[VehicleStatus] = None  # Only active/inactive allowed
    seatingCapacity: Optional[int] = None
    ownerPhone: Optional[str] = None


######## queue check in##########

class QRRequest(BaseModel):
    qr_data: str
    name: str
    contact: str


######## Device Registration Request ##########
class RegisterDeviceRequest(BaseModel):
    device_id: str
    fcm_token: str
    username: str

######## Device Unregistration Request ##########
class UnregisterDeviceRequest(BaseModel):
    device_id: str


class TestFcmNotificationRequest(BaseModel):
    title: str = "Test Notification"
    body: str = "This is a test notification."
