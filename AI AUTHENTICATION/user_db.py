import json
import os

USER_DB_FILE = "users_db.json"

def load_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(db, f)

def register_user(username, name, email, voice_features, signature_features):
    db = load_db()
    db[username] = {
        "name": name,
        "email": email,
        "voice_features": voice_features.tolist(),  # numpy array to list if needed
        "signature_features": signature_features.tolist(),
    }
    save_db(db)

def get_user(username):
    db = load_db()
    user = db.get(username)
    if user:
        user["voice_features"] = np.array(user["voice_features"])
        user["signature_features"] = np.array(user["signature_features"])
    return user
