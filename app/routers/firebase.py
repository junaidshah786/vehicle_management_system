from firebase_admin import credentials, firestore, initialize_app
from app.config.config import firebase_credentials_path

cred = credentials.Certificate(firebase_credentials_path)
initialize_app(cred)
db = firestore.client()
