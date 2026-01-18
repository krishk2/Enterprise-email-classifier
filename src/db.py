
import os
import pymongo
import datetime
import bcrypt
import streamlit as st
from dotenv import load_dotenv

# Load explicitly from root (Local Dev)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def get_secret(key, default=None):
    # Try Streamlit Secrets first (Cloud)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass # Secrets not configured or file missing, fall back to environment
    
    # Fallback to os.environ (Local)
    return os.getenv(key, default)

MONGO_URI = get_secret("MONGO_URI")
DB_NAME = get_secret("DB_NAME", "email_classifier_db")

# Cache connection
client = None
db = None

def get_db():
    global client, db
    if db is not None:
        return db
    try:
        if not MONGO_URI:
            print("❌ ERROR: MONGO_URI is missing or empty!")
            return None
        
        # Masked URI for logs
        masked_uri = MONGO_URI.split('@')[-1] if '@' in MONGO_URI else "NO_CREDENTIALS"
        print(f"🔌 Connecting to MongoDB: ...@{masked_uri} (DB: {DB_NAME})")

        # Fail fast (5s timeout)
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Force a check
        client.admin.command('ping')
        
        db = client[DB_NAME]
        print("✅ Connected to MongoDB!")
        return db
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None

# --- AUTH ---
def create_user(username, password):
    database = get_db()
    if database is None: return False, "DB Connection Failed"
    
    users = database['users']
    if users.find_one({'username': username}):
        return False, "User already exists"
    
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users.insert_one({
        'username': username,
        'password': hashed
    })
    return True, "User created"

def verify_user(username, password):
    database = get_db()
    if database is None: return False
    
    user = database['users'].find_one({'username': username})
    if not user:
        return False
    
    if bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True
    return False

# --- HISTORY ---
def save_classification_event(username, subject, body, category, urgency, confidence):
    database = get_db()
    if database is None: return
    
    history = database['history']
    history.insert_one({
        'username': username,
        'subject': subject,
        'body': body,
        'category': category,
        'urgency': urgency,
        'confidence': confidence,
        'timestamp': datetime.datetime.now()
    })

def get_user_history(username):
    database = get_db()
    if database is None: return []
    
    return list(database['history'].find({'username': username}).sort('timestamp', -1))
