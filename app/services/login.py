from jose import jwt
from app.config.config import SECRET_KEY, ALGORITHM

# JWT setup
SECRET_KEY = SECRET_KEY
ALGORITHM = ALGORITHM


# Utility function to create access token without expiry
def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt