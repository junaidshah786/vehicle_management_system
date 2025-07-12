from jose import jwt


# JWT setup
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"


# Utility function to create access token without expiry
def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt