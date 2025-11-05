from database import init_db
import os
if __name__ == "__main__":
    # Initialize database with URL from environment or default
    db_url = os.getenv("DB_URL", "sqlite:///weather_agent.db")
    init_db(db_url)
    print(f"Database initialized at {db_url}")