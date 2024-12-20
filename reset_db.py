import os
import sqlite3
from database import init_db
from init_timetable import init_timetable_data

def reset_database():
    # Remove existing database
    db_path = 'data/attendance.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize new database
    init_db()
    init_timetable_data()
    
    print("Database reset successfully!")

if __name__ == "__main__":
    reset_database()
