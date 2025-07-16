from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate("./vehiclemanagementsystem-e76c4-firebase-adminsdk-fbsvc-db53f875ef.json")
initialize_app(cred)
db = firestore.client()
