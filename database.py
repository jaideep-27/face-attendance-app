import sqlite3
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'attendance.db')

def init_db():
    """Initialize the database with required tables"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create subjects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                subject_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (subject_id) REFERENCES subjects (id)
            )
        ''')

        conn.commit()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def add_user(user_id, name):
    """Add a new user to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO users (id, name) VALUES (?, ?)', (user_id, name))
        conn.commit()
        logger.info(f"Added user: {name} with ID: {user_id}")
        return True
    except sqlite3.IntegrityError:
        logger.error(f"User ID {user_id} already exists")
        return False
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def add_subject(name):
    """Add a new subject to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO subjects (name) VALUES (?)', (name,))
        conn.commit()
        logger.info(f"Added subject: {name}")
        return True
    except sqlite3.IntegrityError:
        logger.error(f"Subject {name} already exists")
        return False
    except Exception as e:
        logger.error(f"Error adding subject: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def mark_attendance(user_id, subject_name):
    """Mark attendance for a user in a subject"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get subject ID
        cursor.execute('SELECT id FROM subjects WHERE name = ?', (subject_name,))
        result = cursor.fetchone()
        if not result:
            logger.error(f"Subject {subject_name} not found")
            return False
        subject_id = result[0]
        
        # Check if attendance already marked for today
        today = datetime.now().date()
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE user_id = ? AND subject_id = ? 
            AND date(timestamp) = date(?)
        ''', (user_id, subject_id, today))
        
        if cursor.fetchone()[0] > 0:
            logger.info(f"Attendance already marked for user {user_id} in {subject_name} today")
            return True
        
        # Mark attendance
        cursor.execute('''
            INSERT INTO attendance (user_id, subject_id) 
            VALUES (?, ?)
        ''', (user_id, subject_id))
        
        conn.commit()
        logger.info(f"Marked attendance for user {user_id} in {subject_name}")
        return True
    except Exception as e:
        logger.error(f"Error marking attendance: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def get_available_subjects():
    """Get list of all subjects"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name FROM subjects ORDER BY name')
        subjects = [row[0] for row in cursor.fetchall()]
        return subjects
    except Exception as e:
        logger.error(f"Error getting subjects: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def get_attendance_history(date=None):
    """Get attendance history, optionally filtered by date"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = '''
            SELECT u.id, u.name, s.name, a.timestamp
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            JOIN subjects s ON a.subject_id = s.id
        '''
        params = []
        
        if date:
            query += ' WHERE date(a.timestamp) = date(?)'
            params.append(date)
            
        query += ' ORDER BY a.timestamp DESC'
        
        cursor.execute(query, params)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting attendance history: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def get_user_by_id(user_id):
    """Get user details by ID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name, id FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        return result if result else None
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def list_users():
    """List all users in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name FROM users')
        users = cursor.fetchall()
        return users
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def reset_db():
    """Reset the database by removing and reinitializing it"""
    try:
        # Remove existing database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info("Removed existing database")
        
        # Remove features file
        features_file = os.path.join('model', 'features.json')
        if os.path.exists(features_file):
            os.remove(features_file)
            logger.info("Removed features file")
            
        # Remove images directory
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        if os.path.exists(images_dir):
            import shutil
            shutil.rmtree(images_dir)
            logger.info("Removed images directory")
        
        # Initialize fresh database
        return init_db()
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return False