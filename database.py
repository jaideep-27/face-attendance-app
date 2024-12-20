import sqlite3
import os
from datetime import datetime, time
import time as time_module

def get_db():
    """Get a database connection with proper timeout settings"""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        
    conn = sqlite3.connect('data/attendance.db', timeout=20)
    conn.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
    return conn

def init_db():
    """Initialize the database with all required tables"""
    conn = get_db()
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY,
                  name TEXT NOT NULL,
                  roll_number TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS subjects
                 (id INTEGER PRIMARY KEY,
                  name TEXT NOT NULL,
                  code TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS timetable
                 (id INTEGER PRIMARY KEY,
                  subject_id INTEGER,
                  day_of_week INTEGER,
                  start_time TEXT,
                  end_time TEXT,
                  FOREIGN KEY (subject_id) REFERENCES subjects (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  subject_id INTEGER,
                  timestamp TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id),
                  FOREIGN KEY (subject_id) REFERENCES subjects (id))''')
    
    conn.commit()
    conn.close()

def add_user(name, roll_number):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (name, roll_number) VALUES (?, ?)', (name, roll_number))
        user_id = c.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def add_subject(name, code):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO subjects (name, code) VALUES (?, ?)', (name, code))
        subject_id = c.lastrowid
        conn.commit()
        return subject_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def add_timetable_entry(subject_id, day_of_week, start_time, end_time):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO timetable (subject_id, day_of_week, start_time, end_time)
                     VALUES (?, ?, ?, ?)''', (subject_id, day_of_week, start_time, end_time))
        conn.commit()
    finally:
        conn.close()

def mark_attendance(user_id, subject_id):
    """Mark attendance for a user in a subject"""
    conn = get_db()
    c = conn.cursor()
    try:
        # Check if attendance already marked today
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE user_id = ? AND subject_id = ? 
            AND date(timestamp) = ?
        ''', (user_id, subject_id, today))
        
        if c.fetchone()[0] > 0:
            return False
            
        # Mark attendance with current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''
            INSERT INTO attendance (user_id, subject_id, timestamp)
            VALUES (?, ?, ?)
        ''', (user_id, subject_id, timestamp))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False
    finally:
        conn.close()

def get_available_subjects(day_of_week):
    """Get available subjects for the current time"""
    conn = get_db()
    c = conn.cursor()
    try:
        current_time = datetime.now().strftime('%H:%M')
        
        c.execute('''
            SELECT s.id, s.name, s.code, t.start_time, t.end_time
            FROM subjects s
            JOIN timetable t ON s.id = t.subject_id
            WHERE t.day_of_week = ?
            AND t.start_time <= ?
            AND t.end_time >= ?
            ORDER BY t.start_time
        ''', (day_of_week, current_time, current_time))
        
        return c.fetchall()
    finally:
        conn.close()

def get_attendance_history(user_id):
    """Get attendance history for a user"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT s.id, s.name, s.code, a.timestamp
            FROM attendance a
            JOIN subjects s ON a.subject_id = s.id
            WHERE a.user_id = ?
            ORDER BY a.timestamp DESC
        ''', (user_id,))
        return c.fetchall()
    finally:
        conn.close()

def get_user_details(user_id):
    """Get user details by ID"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('SELECT name, roll_number FROM users WHERE id = ?', (user_id,))
        return c.fetchone()
    finally:
        conn.close()